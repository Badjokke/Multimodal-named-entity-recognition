from torch import float32
from transformers import AutoTokenizer, LlamaModel, ViTImageProcessor, ViTModel, BertTokenizerFast, BertModel

from model.configuration.quantization import create_default_quantization_config
from model.language.LSTM import LSTM
from model.language.TransformerCRF import TransformerCRF
from model.multimodal.CrossAttentionMultimodalModel import CrossAttentionModel
from model.multimodal.LinearFusionMultimodalModel import LinearFusionMultimodalModel
from model.multimodal.PartialPredictionMultimodalModel import PartialPredictionMultimodalModel
from model.visual.AlexNetCNN import ConvNet
from model.visual.ViTWrapper import ViT

from model.visual.VisualModelClassifier import VisualModelClassifier


class ModelFactory:

    @staticmethod
    def __create_vit():
        model_name = "google/vit-base-patch16-224-in21k"
        processor = ViTImageProcessor.from_pretrained(model_name)
        model = ViTModel.from_pretrained(model_name)
        return model, processor

    @staticmethod
    def __create_convolutional_net():
        return ConvNet()

    @staticmethod
    def __create_lstm(vocab, bidirectional=True):
        return LSTM(vocab, bidirectional)

    @staticmethod
    def __create_bert_large():
        tokenizer = BertTokenizerFast.from_pretrained('bert-large-uncased', device_map="auto", low_cpu_mem_usage=True,
                                                      torch_dtype=float32)
        model = BertModel.from_pretrained("bert-large-uncased")
        return model, tokenizer

    @staticmethod
    def __create_llama_model_quantized():
        model_name = "meta-llama/Llama-3.1-8B"
        model = LlamaModel.from_pretrained(
            model_name,
            device_map="auto",
            low_cpu_mem_usage=True,
            torch_dtype=float32,
            quantization_config=create_default_quantization_config()
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name, add_bos_token=True)
        tokenizer.pad_token = tokenizer.eos_token
        return model, tokenizer

    """
    -- visual factory functions
    """
    @staticmethod
    def create_vit_classifier(label_count):
        vit, processor = ModelFactory.__create_vit()
        return ModelFactory.__create_visual_model_classificator(ViT(vit, processor), label_count)

    @staticmethod
    def create_cnn_classifier(label_count):
        conv = ModelFactory.__create_convolutional_net()
        return ModelFactory.__create_visual_model_classificator(conv, label_count)
    """
    -- lstm factory functions --
    """
    @staticmethod
    def create_lstm_text_only_classifier(label_count, vocab_size, bidirectional=True):
        lstm, tokenizer = ModelFactory.__create_lstm(vocab_size, bidirectional)
        return TransformerCRF(lstm, label_count), tokenizer

    @staticmethod
    def create_lstm_vit_attention_classifier(label_count, vocab_size, bidirectional=True):
        lstm, tokenizer = ModelFactory.__create_lstm(vocab_size, bidirectional)
        vit, processor = ModelFactory.__create_vit()
        return ModelFactory.__create_cross_attention_multimodal_model(ViT(vit, processor), lstm, label_count), tokenizer

    @staticmethod
    def create_lstm_vit_linear_fusion(label_count, vocab_size, bidirectional=True):
        lstm, tokenizer = ModelFactory.__create_lstm(vocab_size, bidirectional)
        vit, processor = ModelFactory.__create_vit()
        return ModelFactory.__create_linear_fusion_model(lstm, ViT(vit, processor), label_count), tokenizer

    @staticmethod
    def create_bert_vit_partial_prediction(label_count, vocab_size, bidirectional=True):
        lstm, tokenizer = ModelFactory.__create_lstm(vocab_size, bidirectional)
        vit, processor = ModelFactory.__create_vit()
        return ModelFactory.__create_partial_prediction_model(lstm, ViT(vit, processor), label_count), tokenizer

    """
    -- bert factory functions --
    """
    @staticmethod
    def create_bert_text_only_classifier(label_count):
        bert, tokenizer = ModelFactory.__create_bert_large()
        return TransformerCRF(bert, label_count), tokenizer

    @staticmethod
    def create_bert_vit_attention_classifier(label_count):
        bert, tokenizer = ModelFactory.__create_bert_large()
        vit, processor = ModelFactory.__create_vit()
        return ModelFactory.__create_cross_attention_multimodal_model(ViT(vit, processor), bert,label_count), tokenizer

    @staticmethod
    def create_bert_vit_linear_fusion(label_count):
        bert, tokenizer = ModelFactory.__create_bert_large()
        vit, processor = ModelFactory.__create_vit()
        return ModelFactory.__create_linear_fusion_model(bert, ViT(vit, processor), label_count), tokenizer

    @staticmethod
    def create_bert_vit_partial_prediction(label_count):
        bert, tokenizer = ModelFactory.__create_bert_large()
        vit, processor = ModelFactory.__create_vit()
        return ModelFactory.__create_partial_prediction_model(bert, ViT(vit, processor), label_count), tokenizer

    """
    -- llama factory functions --
    """
    @staticmethod
    def create_llama_text_only_classifier(label_count):
        llama, tokenizer = ModelFactory.__create_llama_model_quantized()
        return TransformerCRF(llama, label_count), tokenizer

    @staticmethod
    def create_llama_vit_attention_classifier(label_count):
        llama, tokenizer = ModelFactory.__create_llama_model_quantized()
        vit, processor = ModelFactory.__create_vit()
        return ModelFactory.__create_cross_attention_multimodal_model(ViT(vit, processor), llama, label_count), tokenizer

    @staticmethod
    def create_llama_vit_linear_fusion(label_count):
        llama, tokenizer = ModelFactory.__create_llama_model_quantized()
        vit, processor = ModelFactory.__create_vit()
        return ModelFactory.__create_linear_fusion_model(llama, ViT(vit, processor), label_count), tokenizer

    @staticmethod
    def create_llama_vit_partial_prediction(label_count):
        llama, tokenizer = ModelFactory.__create_llama_model_quantized()
        vit, processor = ModelFactory.__create_vit()
        return ModelFactory.__create_partial_prediction_model(llama, ViT(vit, processor), label_count), tokenizer

    """
    -- generic models factory functions --
    """
    @staticmethod
    def __create_cross_attention_multimodal_model(text_model, visual_model, label_count):
        return CrossAttentionModel(text_model, visual_model, label_count)

    @staticmethod
    def __create_linear_fusion_model(text_model, visual_model, label_count):
        return LinearFusionMultimodalModel(text_model, visual_model, label_count)

    @staticmethod
    def __create_partial_prediction_model(text_model, visual_model, label_count):
        return PartialPredictionMultimodalModel(text_model, visual_model, label_count)

    @staticmethod
    def __create_visual_model_classificator(visual_model, label_count):
        return VisualModelClassifier(visual_model, label_count)
