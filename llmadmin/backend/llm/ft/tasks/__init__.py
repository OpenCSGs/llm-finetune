from . import sequenceclassification_glue_cola
from . import sequenceclassification_glue_mrpc
from . import tokenclassification_conll2003
from . import noheader_AdvertiseGen
from . import text_generation_AdvertiseGen
from . import maskedlm_imdb
from . import sequenceclassification_yelp_review_full

TASK_REGISTRY = {
    "sequenceclassification-glue-cola": sequenceclassification_glue_cola.SequenceclassificationGlueCola,
    "sequenceclassification-glue-mrpc": sequenceclassification_glue_mrpc.SequenceclassificationGlueMrpc,
    "tokenclassification-conll2003": tokenclassification_conll2003.TokenclassificationConll2003,
    "noheader-AdvertiseGen": noheader_AdvertiseGen.NoheaderAdvertiseGen,
    "text-generation-AdvertiseGen": text_generation_AdvertiseGen.NoheaderAdvertiseGen,
    "maskedlm-imdb": maskedlm_imdb.MaskedLMImdb,
    "sequenceclassification-yelp_review_full": sequenceclassification_yelp_review_full.SequenceclassificationYelpReviewFull
}