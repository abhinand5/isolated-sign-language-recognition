import os
import tensorflow as tf


class TFLiteModel(tf.Module):
    def __init__(self, model, preprocess_layer):
        super(TFLiteModel, self).__init__()

        # Load the feature generation and main models
        self.preprocess_layer = preprocess_layer
        self.model = model

    @tf.function(
        input_signature=[
            tf.TensorSpec(shape=[None, 543, 3], dtype=tf.float32, name="inputs")
        ]
    )
    def __call__(self, inputs):
        # Preprocess Data
        x, non_empty_frame_idxs = self.preprocess_layer(inputs)
        # Add Batch Dimension
        x = tf.expand_dims(x, axis=0)
        non_empty_frame_idxs = tf.expand_dims(non_empty_frame_idxs, axis=0)
        # Make Prediction
        outputs = self.model(
            {"frames": x, "non_empty_frame_idxs": non_empty_frame_idxs}
        )
        # Squeeze Output 1x250 -> 250
        outputs = tf.squeeze(outputs, axis=0)

        # Return a dictionary with the output tensor
        return {"outputs": outputs}


# TFLite ensemble model for submission
class TFLiteEnsembleModel(tf.Module):
    def __init__(self, models, preprocess_layer):
        super(TFLiteEnsembleModel, self).__init__()

        # Load the feature generation and main models
        self.preprocess_layer = preprocess_layer
        self.models = models

    @tf.function(
        input_signature=[
            tf.TensorSpec(shape=[None, 543, 3], dtype=tf.float32, name="inputs")
        ]
    )
    def __call__(self, inputs):
        # Preprocess the inputs
        x, non_empty_frame_idxs = self.proprocess_inputs(inputs)

        outputs = []
        # Make Prediction
        for _model in self.models:
            output = _model({"frames": x, "non_empty_frame_idxs": non_empty_frame_idxs})
            outputs.append(output)

        outputs = tf.concat(outputs, axis=0)
        outputs = tf.reduce_mean(outputs, axis=0, keepdims=True)

        # Squeeze Output 1x250 -> 250
        outputs = tf.squeeze(outputs, axis=0)

        # Return a dictionary with the output tensor
        return {"outputs": outputs}

    def proprocess_inputs(self, inputs):
        # Preprocess Data
        x, non_empty_frame_idxs = self.preprocess_layer(inputs)
        # Add Batch Dimension
        x = tf.expand_dims(x, axis=0)
        non_empty_frame_idxs = tf.expand_dims(non_empty_frame_idxs, axis=0)

        return x, non_empty_frame_idxs


def convert_to_tflite(
    name, tflite_keras_model, model_dir, quantize_model=False, quant_method="dynamic"
):
    if not quantize_model:
        # Create Model Converter
        keras_model_converter = tf.lite.TFLiteConverter.from_keras_model(
            tflite_keras_model
        )
        # Convert Model
        tflite_model = keras_model_converter.convert()
        # Write Model
        with open(os.path.join(model_dir, f"{name}.tflite"), "wb") as f:
            f.write(tflite_model)
    else:
        if quant_method == "float16":
            # Alternative 16-bit float quantization
            # It reduces model size by up to half
            # But the runtime remains more or less the same
            keras_model_quant_converter = tf.lite.TFLiteConverter.from_keras_model(
                tflite_keras_model
            )
            keras_model_quant_converter.optimizations = [tf.lite.Optimize.DEFAULT]
            keras_model_quant_converter.target_spec.supported_types = [tf.float16]
            tflite_quant_model = keras_model_quant_converter.convert()

            with open("/kaggle/working/model-quant.tflite", "wb") as f:
                f.write(tflite_quant_model)

        elif quant_method == "dynamic":
            keras_model_quant_converter = tf.lite.TFLiteConverter.from_keras_model(
                tflite_keras_model
            )
            keras_model_quant_converter.optimizations = [tf.lite.Optimize.DEFAULT]
            tflite_quant_model = keras_model_quant_converter.convert()

            with open(os.path.join(model_dir, f"{name}-quant.tflite"), "wb") as f:
                f.write(tflite_quant_model)
        else:
            raise ValueError(f"Unsupported value: {quant_method}")
