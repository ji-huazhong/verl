_BATCH_HDP_GROUP = None


def get_batch_hdp_group():
    global _BATCH_HDP_GROUP
    return _BATCH_HDP_GROUP


def set_batch_hdp_group(batch_hdp_group):
    global _BATCH_HDP_GROUP
    _BATCH_HDP_GROUP = batch_hdp_group


def clean_hdp_group():
    global _BATCH_HDP_GROUP
    _BATCH_HDP_GROUP = None
