Role-aware disk offload
=======================

Last updated: 08/25/2026.

Why disk offload?
-----------------

Colocated RL improves accelerator utilization by time-sharing the same GPUs
between training and inference roles. This requires inactive roles to vacate
HBM at phase boundaries. During rollout, the actor's training state can be
offloaded; during actor training, rollout-engine weights and an inactive
reference policy may need to be offloaded so the actor can use the GPUs.
This feature applies to training-engine state owned by actor, reference, and
critic workers. It does not replace the rollout engine's own sleep or
weight-release mechanism.

CPU offload is the preferred first tier because it has lower latency than
storage, but it shifts the capacity pressure from HBM to host memory. Each rank
copies its local model shards or replicas and optimizer state, and several
inactive roles may be resident in host memory at the same time. On servers
whose host memory has not grown in proportion to their aggregate accelerator
memory, the actor training phase can therefore encounter a host OOM even though
enough HBM has been reclaimed. As model state and per-node accelerator capacity
continue to grow, provisioning DRAM at the same rate also becomes increasingly
expensive, making this constraint more common.

Disk offload adds node-local NVMe as a capacity tier for this case. It avoids
retaining a full user-space CPU copy of a component by moving state through a
pair of reusable staging buffers. ``chunk_size_mb`` controls the size of each
buffer. An engine store allocates these buffers lazily when disk I/O first
occurs and can then retain up to ``2 * chunk_size_mb`` of staging memory.
Operating-system page cache and fallback read allocations on platforms without
``preadv`` are additional and are not bounded by this setting. The trade-off is
additional latency at phase transitions and additional storage traffic.

Disk offload configuration
--------------------------

Disk offload is configured per role and for each state type exposed by that
backend. Optimizer state is typically the best candidate because it is large
and inactive outside the optimizer step. The following example moves actor
parameters and gradients to CPU and Megatron optimizer state to disk.

.. code-block:: yaml

   actor_rollout_ref:
     actor:
       megatron:
         offload:
           param:
             target: cpu
           grad:
             target: cpu
           optimizer:
             target: disk
           disk:
             path: /local_nvme/verl-offload
             chunk_size_mb: 64
             cleanup_on_exit: true

Each component accepts ``none``, ``cpu``, or ``disk`` when its backend supports
that target. Parameter and gradient disk targets remain available for jobs that
cannot fit those inactive states in host memory. ``offload.disk.path`` is
required when any component selects ``disk``.

Megatron and VeOmni reference parameters follow the actor's parameter target
and disk settings unless explicitly overridden. FSDP references retain their
pre-existing forward-only CPU offload when ``param.target`` is ``null``;
``none`` disables that implicit behavior, while ``cpu`` and ``disk`` select an
explicit target. Critic offload settings are independent of the actor and must
be configured explicitly.

``null`` is a compatibility sentinel rather than an offload target: it allows
the legacy boolean or backend default to decide the effective policy. Use
``none`` to explicitly disable offload.

For backward compatibility, each backend's existing boolean fields remain
available temporarily. Megatron retains ``param_offload``, ``grad_offload``,
and ``optimizer_offload``; other backends retain only the fields they already
exposed. They emit a ``FutureWarning`` and map ``true`` to ``cpu`` and ``false``
to ``none``. A legacy ``true`` cannot be combined with an explicit target for
the same component. Although an explicit target takes precedence over a legacy
``false``, new configurations should not mix the two forms.

Disk-target support matrix
--------------------------

.. list-table::
   :header-rows: 1

   * - Backend
     - Disk param
     - Disk grad
     - Disk optimizer
     - Result when configured
   * - Megatron
     - yes
     - yes
     - yes
     - supported
   * - FSDP1 / FSDP2
     - yes
     - implicit with param
     - yes
     - supported
   * - VeOmni
     - yes
     - implicit with param
     - yes
     - supported
   * - Megatron-FSDP
     - TBD
     - TBD
     - TBD
     - ``AssertionError`` during config validation
   * - MindSpeed / NPU
     - yes
     - yes
     - yes
     - supported through Megatron
   * - AutoModel
     - TBD
     - TBD
     - TBD
     - ``AssertionError`` during config validation
   * - FSDP Turbo / TorchTitan
     - TBD
     - TBD
     - TBD
     - ``AssertionError`` during config validation

``TBD`` means that disk offload is not supported by the current implementation
but may be added in a future release. Configuring a TBD target currently raises
the error shown in the final column rather than silently ignoring the target.

This matrix covers ``target: disk`` only. FSDP1/FSDP2 and VeOmni do not expose
``grad.target`` because their original offload interface did not expose an
independent gradient switch. Gradient storage follows parameter placement, so
``param.target: disk`` also serializes live gradients to disk and
``param.target: cpu`` keeps the existing CPU behavior. TorchTitan likewise
moves gradient storage with parameters for CPU offload and rejects disk targets.

FSDP and VeOmni also reject combining disk targets with ``offload_policy`` and
``enable_fsdp_offload``, respectively. Invalid targets and unsupported
combinations follow verl's existing configuration style and are checked with
``assert``.

Disk layout and memory use
--------------------------

``offload.disk.path`` must point to fast node-local storage. verl creates an
isolated scratch directory for every engine store, so colocated roles may share
the same configured root without colliding. In a multi-node job, the same path
must resolve to local storage on every node. For each component, a store uses
one reusable flat data file plus a manifest and generation marker; it does not
create one file per tensor.

Disk I/O is chunked and double-buffered. The public store call remains
synchronous, but internally one pinned CPU buffer can perform file I/O while
the other transfers the adjacent chunk between host and accelerator on a
dedicated copy stream. Reads use ``preadv`` where available to fill the staging
buffer directly. ``chunk_size_mb`` bounds each buffer, so one active store can
retain up to ``2 * chunk_size_mb`` of staging memory without retaining a full
user-space copy of the component. Operating-system page cache remains outside
this bound. Accelerator storage is released only after the complete disk
generation has been written and published for the current process; the files
are not made crash-durable. ``cleanup_on_exit`` uses a Python exit handler and
removes only the exact store directory carrying the store's ownership marker.
Cleanup is best effort: abrupt worker or node termination can leave scratch
directories behind.

Provision enough capacity for the rank-local state of every disk-target
component and engine store on a node. Parameter, gradient, and optimizer files
coexist, and colocated actor, reference, and critic stores are independent.
The files are reused across phase transitions, but no cluster-wide capacity
check runs before the first write. Production deployments should monitor free
space and remove orphaned job directories according to their retention policy.

FSDP1 flat parameters and FSDP2/VeOmni DTensors are restored in place. verl
writes each unique rank-local backing storage once, including shared flat-buffer
storage, then resizes that same storage to zero. Onload expands and refills the
same object, preserving Parameter identity, DTensor placements, and aliases.

Gradient semantics
------------------

Gradient data is written only while it is live.  This matters for split
training APIs that separate ``forward_backward`` from ``optimizer_step`` (for
example, the Tinker worker): leaving the first call with
``zero_grad_on_exit=false`` persists the gradient, and entering the optimizer
step restores it.

In the standard PPO/GRPO update path, the optimizer has already consumed the
gradient and the train context clears it before offload.  verl then applies its
existing gradient-buffer reclamation and does not write cleared gradients to
disk. Cleared gradients are not restored from disk: Megatron recreates its
gradient buffers before use, while FSDP and VeOmni allow autograd to recreate
``param.grad`` during the next backward pass.

Disk offload limitations
------------------------

* Disk offload for Megatron-FSDP, AutoModel, FSDP Turbo, and TorchTitan is TBD.
  The current implementation rejects disk targets.
* Disk offload reclaims inactive state only. Each component is restored before
  use, so it does not reduce the memory peak of an active forward, backward, or
  ``optimizer.step``.
* The path is scratch storage, not a checkpoint.  It is not portable across
  jobs or distributed topologies, and writes are not synchronized to durable
  media before accelerator storage is released.
* Store calls wait for all staged copies and file I/O to complete. Cross-phase
  asynchronous offload and prefetch are not implemented.
* Configuration validation uses ``assert`` to match existing verl engine
  configuration style and is disabled when Python runs with ``-O``.
* Use local NVMe.  Shared filesystems can create severe rank-wide tail latency.
* ``cleanup_on_exit`` cannot clean files after abrupt process or node
  termination; operators should provide orphan cleanup for the configured
  scratch root.
* Checkpoints temporarily restore the selected model and optimizer state and
  continue to use the existing checkpoint format.
