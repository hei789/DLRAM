from transformers import AutoModel, AutoTokenizer

# 下载 BERT 模型、Tokenizer
bert = AutoModel.from_pretrained("bert-base-uncased")
bert.save_pretrained("./models/bert-base-uncased")

bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
bert_tokenizer.save_pretrained("./models/bert-base-uncased")

# 下载 ViT 模型、Image Processor
try:
    # transformers >= 4.25.0
    from transformers import AutoImageProcessor
    vit_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
except ImportError:
    # transformers < 4.25.0 使用 AutoFeatureExtractor
    from transformers import AutoFeatureExtractor
    vit_processor = AutoFeatureExtractor.from_pretrained("google/vit-base-patch16-224")

vit = AutoModel.from_pretrained("google/vit-base-patch16-224")
vit.save_pretrained("./models/vit-base-patch16-224")
vit_processor.save_pretrained("./models/vit-base-patch16-224")
