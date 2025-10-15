import os
import json
import math
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, Model, Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import uuid
from tensorflow.keras import backend as K
import cv2 # Import OpenCV for CAM generation

# ==== Paths ====
base_dir = r'C:\Users\Abhay\Desktop\Lungs disease AI and ML\Scans_data4\data'
train_dir = os.path.join(base_dir, 'train')
test_dir = os.path.join(base_dir, 'test')

# ==== Constants ====
img_size = (224, 224)
batch_size = 32
initial_epochs = 15
fine_tune_epochs = 10

@tf.keras.utils.register_keras_serializable()
# ==== Enhanced Components with get_config methods ====
class VisionTransformerBlock(layers.Layer):
    """Vision Transformer block with multi-head attention"""
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, **kwargs):
        super(VisionTransformerBlock, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.rate = rate
        
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        
        # Fix: Create named layers instead of Sequential
        self.ffn_dense1 = layers.Dense(ff_dim, activation="relu", name="ffn_dense1")
        self.ffn_dense2 = layers.Dense(embed_dim, name="ffn_dense2")
        
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training=None):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        
        # Fix: Use individual layers instead of Sequential
        ffn_output = self.ffn_dense1(out1)
        ffn_output = self.ffn_dense2(ffn_output)
        
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'embed_dim': self.embed_dim,
            'num_heads': self.num_heads,
            'ff_dim': self.ff_dim,
            'rate': self.rate
        })
        return config

# ==== Fix for the reshape_patches serialization issue ====

@tf.keras.utils.register_keras_serializable()
class ReshapePatchesLayer(layers.Layer):
    """Custom layer to reshape patches for Vision Transformer"""
    def __init__(self, embed_dim, **kwargs):
        super(ReshapePatchesLayer, self).__init__(**kwargs)
        self.embed_dim = embed_dim
    
    def call(self, x):
        batch_size = tf.shape(x)[0]
        height, width = tf.shape(x)[1], tf.shape(x)[2]
        patch_dims = height * width
        return tf.reshape(x, [batch_size, patch_dims, self.embed_dim])
    
    def get_config(self):
        config = super().get_config()
        config.update({'embed_dim': self.embed_dim})
        return config

@tf.keras.utils.register_keras_serializable()
class PositionalEncodingLayer(layers.Layer):
    """Custom layer to add positional encoding - FIXED VERSION"""
    def __init__(self, max_patches, embed_dim, **kwargs):
        super(PositionalEncodingLayer, self).__init__(**kwargs)
        self.max_patches = max_patches
        self.embed_dim = embed_dim
    
    def build(self, input_shape):
        # Create positional embeddings as layer weights for better serialization
        self.pos_embedding = self.add_weight(
            name='positional_embedding',
            shape=(self.max_patches, self.embed_dim),
            initializer='uniform',
            trainable=True
        )
        super().build(input_shape)
    
    def call(self, x):
        seq_len = tf.shape(x)[1]
        # Use the learned positional embeddings
        pos_emb = self.pos_embedding[:seq_len, :]
        return x + pos_emb
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'max_patches': self.max_patches,
            'embed_dim': self.embed_dim
        })
        return config

def create_vision_transformer_hybrid(cnn_features, patch_size=7, embed_dim=256, num_heads=8):
    """Hybrid CNN-ViT architecture - FIXED VERSION"""
    patches = layers.Conv2D(embed_dim, kernel_size=patch_size, strides=patch_size, 
                           padding='valid', name='patch_extraction')(cnn_features)
    
    # Use custom layer instead of Lambda
    patches_reshaped = ReshapePatchesLayer(embed_dim, name='reshape_patches')(patches)
    
    max_patches = 64
    
    # Use custom layer instead of Lambda
    encoded_patches = PositionalEncodingLayer(max_patches, embed_dim, name='add_pos_encoding')(patches_reshaped)
    
    for i in range(3):
        encoded_patches = VisionTransformerBlock(embed_dim, num_heads, embed_dim * 2, name=f'vit_block_{i}')(encoded_patches)
    
    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = layers.GlobalAveragePooling1D()(representation)
    
    return representation

@tf.keras.utils.register_keras_serializable()
class AdaptiveMultiScaleAttention(layers.Layer):
    """Advanced multi-scale attention with adaptive weighting"""
    def __init__(self, num_scales=4, base_filters=64, **kwargs):
        super(AdaptiveMultiScaleAttention, self).__init__(**kwargs)
        self.num_scales = num_scales
        self.base_filters = base_filters
        
        self.scale_convs = []
        self.scale_attentions = []
        for i in range(num_scales):
            kernel_size = 2**i + 1
            self.scale_convs.append(
                layers.Conv2D(base_filters, kernel_size=kernel_size, padding='same', 
                             activation='relu', name=f'scale_conv_{i}')
            )
            self.scale_attentions.append(
                layers.Conv2D(1, 1, activation='sigmoid', name=f'scale_attention_{i}')
            )
        
        self.fusion_conv = layers.Conv2D(base_filters, 1, activation='relu')
        self.channel_attention = layers.Conv2D(base_filters, 1, activation='sigmoid')
        
    def call(self, inputs):
        scale_features = []
        attention_maps = []
        
        for i in range(self.num_scales):
            scale_feat = self.scale_convs[i](inputs)
            attention = self.scale_attentions[i](scale_feat)
            attention_maps.append(attention)
            weighted_feat = layers.Multiply()([scale_feat, attention])
            scale_features.append(weighted_feat)
        
        concatenated = layers.Concatenate()(scale_features)
        fused = self.fusion_conv(concatenated)
        channel_weights = self.channel_attention(fused)
        output = layers.Multiply()([fused, channel_weights])
        
        return output, attention_maps
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'num_scales': self.num_scales,
            'base_filters': self.base_filters
        })
        return config

@tf.keras.utils.register_keras_serializable()
class BayesianDense(layers.Layer):
    """Bayesian Dense layer for uncertainty estimation"""
    def __init__(self, units, prior_sigma=1.0, **kwargs):
        super(BayesianDense, self).__init__(**kwargs)
        self.units = units
        self.prior_sigma = prior_sigma
        
    def build(self, input_shape):
        self.w_mu = self.add_weight(shape=(input_shape[-1], self.units),
                                   initializer='glorot_uniform',
                                   trainable=True, name='w_mu')
        self.w_rho = self.add_weight(shape=(input_shape[-1], self.units),
                                    initializer=tf.keras.initializers.Constant(-3.0),
                                    trainable=True, name='w_rho')
        
        self.b_mu = self.add_weight(shape=(self.units,),
                                   initializer='zeros',
                                   trainable=True, name='b_mu')
        self.b_rho = self.add_weight(shape=(self.units,),
                                    initializer=tf.keras.initializers.Constant(-3.0),
                                    trainable=True, name='b_rho')
        
    def call(self, inputs, training=None):
        w_sigma = tf.nn.softplus(self.w_rho) + 1e-5
        w = self.w_mu + w_sigma * tf.random.normal(tf.shape(self.w_mu))
        
        b_sigma = tf.nn.softplus(self.b_rho) + 1e-5
        b = self.b_mu + b_sigma * tf.random.normal(tf.shape(self.b_mu))
        
        return tf.matmul(inputs, w) + b
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'units': self.units,
            'prior_sigma': self.prior_sigma
        })
        return config

def create_uncertainty_head(features, num_classes):
    """Creates uncertainty-aware prediction head with enhanced stability"""
    # Main prediction path with batch normalization
    x = layers.Dense(256)(features)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Dropout(0.5)(x)
    
    x = BayesianDense(128, name='bayesian_dense_main')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Dropout(0.3)(x)
    
    main_pred = layers.Dense(num_classes, activation='softmax', name='main_predictions')(x)
    
    # Uncertainty estimation path with batch normalization
    unc_x = layers.Dense(128)(features)
    unc_x = layers.BatchNormalization()(unc_x)
    unc_x = layers.ReLU()(unc_x)
    unc_x = layers.Dropout(0.3)(unc_x)
    
    epistemic_unc = layers.Dense(1, activation='sigmoid', name='epistemic_uncertainty')(unc_x)
    
    # Predictive entropy for aleatoric uncertainty
    entropy_features = layers.Dense(64)(features)
    entropy_features = layers.BatchNormalization()(entropy_features)
    entropy_features = layers.ReLU()(entropy_features)
    aleatoric_unc = layers.Dense(1, activation='sigmoid', name='aleatoric_uncertainty')(entropy_features)
    
    return main_pred, epistemic_unc, aleatoric_unc

@tf.keras.utils.register_keras_serializable()
class AnatomicalReasoningModule(layers.Layer):
    """Graph-inspired reasoning for anatomical relationships"""
    def __init__(self, num_regions=16, feature_dim=256, **kwargs):
        super(AnatomicalReasoningModule, self).__init__(**kwargs)
        self.num_regions = num_regions
        self.feature_dim = feature_dim
        
        self.region_conv = layers.Conv2D(feature_dim, 1, activation='relu', name='region_conv')
        self.projection_dense = layers.Dense(feature_dim, activation='relu', name='patch_projection')
        self.multi_head_attention = layers.MultiHeadAttention(
            num_heads=8, 
            key_dim=feature_dim // 8,
            name='anatomical_attention'
        )
        self.global_pooling = layers.GlobalAveragePooling1D(name='global_reasoning_pool')
        
    def call(self, feature_maps):
        region_features = self.region_conv(feature_maps)
        pooled_features = layers.GlobalAveragePooling2D()(region_features)
        
        batch_size = tf.shape(pooled_features)[0]
        sequence_features = tf.expand_dims(pooled_features, axis=1)
        sequence_features = tf.tile(sequence_features, [1, self.num_regions, 1])
        
        positions = tf.range(self.num_regions, dtype=tf.float32)
        pos_embeddings = tf.sin(positions[:, None] / tf.pow(10000.0, 
                               tf.range(0, self.feature_dim, 2, dtype=tf.float32) / self.feature_dim))
        pos_embeddings = tf.tile(pos_embeddings[None, :, :], [batch_size, 1, 1])
        
        if pos_embeddings.shape[-1] != sequence_features.shape[-1]:
            pos_embeddings = self.projection_dense(pos_embeddings)
        
        enhanced_features = sequence_features + pos_embeddings
        reasoned_features = self.multi_head_attention(enhanced_features, enhanced_features)
        global_reasoning = self.global_pooling(reasoned_features)
        
        return global_reasoning
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'num_regions': self.num_regions,
            'feature_dim': self.feature_dim
        })
        return config

@tf.keras.utils.register_keras_serializable()
class AdaptiveFocalLoss(tf.keras.losses.Loss):
    """Advanced focal loss with adaptive gamma"""
    def __init__(self, alpha=0.25, gamma_base=2.0, reduction='sum_over_batch_size', name='adaptive_focal_loss'):
        super(AdaptiveFocalLoss, self).__init__(reduction=reduction, name=name)
        self.alpha = alpha
        self.gamma_base = gamma_base
        
    def call(self, y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        ce_loss = -tf.reduce_sum(y_true * tf.math.log(y_pred), axis=-1)
        p_t = tf.reduce_sum(y_true * y_pred, axis=-1)
        adaptive_gamma = self.gamma_base * (1 - p_t)
        alpha_t = self.alpha
        focal_loss = alpha_t * tf.pow(1 - p_t, adaptive_gamma) * ce_loss
        return focal_loss
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'alpha': self.alpha,
            'gamma_base': self.gamma_base
        })
        return config

@tf.keras.utils.register_keras_serializable()
class SEBlock(layers.Layer):
    """Squeeze-and-Excitation block"""
    def __init__(self, reduction=16, **kwargs):
        super(SEBlock, self).__init__(**kwargs)
        self.reduction = reduction
        
    def build(self, input_shape):
        self.global_avg_pool = layers.GlobalAveragePooling2D(keepdims=True)
        self.dense1 = layers.Dense(input_shape[-1] // self.reduction, activation='relu')
        self.dense2 = layers.Dense(input_shape[-1], activation='sigmoid')
        super().build(input_shape)
        
    def call(self, inputs):
        squeeze = self.global_avg_pool(inputs)
        squeeze = layers.Reshape((1, 1, inputs.shape[-1]))(squeeze)
        excitation = self.dense1(squeeze)
        excitation = self.dense2(excitation)
        return layers.Multiply()([inputs, excitation])
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'reduction': self.reduction
        })
        return config

def residual_block_with_se(x, filters, stride=1, use_se=True):
    """Enhanced residual block with SE attention"""
    shortcut = x
    
    x = layers.Conv2D(filters, 3, strides=stride, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(filters, 3, strides=1, padding='same')(x)
    x = layers.BatchNormalization()(x)
    
    if use_se:
        unique_id = uuid.uuid4().hex[:6]
        x = SEBlock(name=f'se_block_filters_{filters}_stride_{stride}_{unique_id}')(x)
    
    if stride != 1 or shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, 1, strides=stride, padding='same')(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)
    
    x = layers.Add()([x, shortcut])
    x = layers.ReLU()(x)
    return x

def build_enhanced_backbone():
    """Enhanced backbone with multiple novel components"""
    img_input = Input(shape=(224, 224, 3), name='image_input')
    
    x = layers.Conv2D(64, 7, strides=2, padding='same', activation='relu')(img_input)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(3, strides=2, padding='same')(x)
    
    x = residual_block_with_se(x, 64, stride=1)
    x = residual_block_with_se(x, 64, stride=1)
    
    x = residual_block_with_se(x, 128, stride=2)
    x = residual_block_with_se(x, 128, stride=1)
    
    x = residual_block_with_se(x, 256, stride=2)
    x = residual_block_with_se(x, 256, stride=1)
    
    amsa = AdaptiveMultiScaleAttention(num_scales=4, base_filters=128, name='adaptive_multiscale_attention')
    attended_features, attention_maps = amsa(x)
    
    x = residual_block_with_se(attended_features, 512, stride=2)
    x = residual_block_with_se(x, 512, stride=1)
    
    return img_input, x, attention_maps

def build_medifusion_enhanced():
    """Build the complete MediFusion-Net Enhanced model"""
    img_input, cnn_features, attention_maps = build_enhanced_backbone()
    
    vit_features = create_vision_transformer_hybrid(cnn_features, patch_size=7, embed_dim=256, num_heads=8)
    
    anatomical_reasoning = AnatomicalReasoningModule(num_regions=16, feature_dim=256, name='anatomical_reasoning')
    reasoning_features = anatomical_reasoning(cnn_features)
    
    cnn_global = layers.GlobalAveragePooling2D()(cnn_features)
    combined_features = layers.Concatenate()([cnn_global, vit_features, reasoning_features])
    
    refined_features = layers.Dense(1024, activation='relu')(combined_features)
    refined_features = layers.Dropout(0.5)(refined_features)
    refined_features = layers.Dense(512, activation='relu')(refined_features)
    refined_features = layers.Dropout(0.3)(refined_features)
    
    main_predictions, epistemic_unc, aleatoric_unc = create_uncertainty_head(refined_features, num_classes=3)
    
    attention_features = layers.GlobalAveragePooling2D()(attention_maps[0])
    attention_output = layers.Dense(64, activation='relu', name='attention_features')(attention_features)
    
    model = Model(
        inputs=img_input,
        outputs={
            'main_predictions': main_predictions,
            'epistemic_uncertainty': epistemic_unc,
            'aleatoric_uncertainty': aleatoric_unc,
            'attention_features': attention_output
        }
    )
    
    return model

def create_medical_augmentation():
    """Medical-specific data augmentation - FIXED VERSION"""
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.05,
        zoom_range=0.1,
        horizontal_flip=True,
        brightness_range=[0.9, 1.1],
        channel_shift_range=0.05,
        fill_mode='nearest'
        # Remove the lambda function for better serialization
    )
    
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    def multi_output_generator(generator):
        while True:
            batch_x, batch_y = next(generator)
            # Apply noise augmentation here if needed during training
            if generator == train_base:  # Only for training
                noise = np.random.normal(0, 0.01, batch_x.shape)
                batch_x = batch_x + noise
                batch_x = np.clip(batch_x, 0, 1)  # Ensure values stay in [0,1]
            
            yield batch_x, {
                'main_predictions': batch_y,
                'epistemic_uncertainty': np.zeros((len(batch_y), 1)),
                'aleatoric_uncertainty': np.zeros((len(batch_y), 1)),
                'attention_features': np.zeros((len(batch_y), 64))
            }

    train_base = train_datagen.flow_from_directory(
        train_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True
    )
    
    test_base = test_datagen.flow_from_directory(
        test_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )

    train_generator = multi_output_generator(train_base)
    test_generator = multi_output_generator(test_base)
    
    return train_generator, test_generator, train_base, test_base

class EnhancedModelCheckpoint(ModelCheckpoint):
    """Enhanced model checkpoint with stability checks"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.best_weights = None
        
    def on_epoch_end(self, epoch, logs=None):
        current = logs.get(self.monitor)
        if current is None:
            return
            
        if self.best_weights is None or self.monitor_op(current, self.best):
            self.best = current
            self.best_weights = self.model.get_weights()
            self.model.save(self.filepath, overwrite=True)

class UncertaintyMetrics(tf.keras.callbacks.Callback):
    """Custom callback to track uncertainty metrics"""
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        if 'val_main_predictions_accuracy' in logs and 'val_main_predictions_loss' in logs:
            if logs['val_main_predictions_loss'] > 1.5 * logs['val_main_predictions_accuracy']:
                print("\nWarning: High loss relative to accuracy - possible training instability")
                

def perform_uncertainty_calibration(predictions, true_labels):
    """Perform uncertainty calibration analysis"""
    main_preds = predictions['main_predictions']
    predicted_labels = np.argmax(main_preds, axis=1)
    confidence = np.max(main_preds, axis=1)
    correct = (predicted_labels == true_labels).astype(float)

    # Bin predictions by confidence
    bin_boundaries = np.linspace(0, 1, 11)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    calibration_data = []
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidence > bin_lower) & (confidence <= bin_upper)
        prop_in_bin = in_bin.mean()

        if prop_in_bin > 0:
            accuracy_in_bin = correct[in_bin].mean()
            avg_confidence_in_bin = confidence[in_bin].mean()
            calibration_data.append({
                'bin_lower': bin_lower,
                'bin_upper': bin_upper,
                'accuracy': accuracy_in_bin,
                'confidence': avg_confidence_in_bin,
                'prop_in_bin': prop_in_bin
            })

    return calibration_data

def plot_calibration_curve(calibration_data):
    """Plot calibration curve"""
    if not calibration_data:
        print("No calibration data available")
        return

    confidences = [item['confidence'] for item in calibration_data]
    accuracies = [item['accuracy'] for item in calibration_data]

    plt.figure(figsize=(8, 6))
    plt.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
    plt.plot(confidences, accuracies, 'bo-', label='Model calibration')
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('Fraction of Positives')
    plt.title('Calibration Plot')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('calibration_curve.png', dpi=300, bbox_inches='tight')
    plt.show()
    
# ==== Additional Helper Functions ====
def generate_class_activation_maps(model, test_base, class_names, num_samples=5):
    """Generate Class Activation Maps for model interpretability"""
    print("\nGenerating Class Activation Maps...")

    # Reset the base test generator
    test_base.reset()
    
    # Get a batch of test images from the base generator
    batch_x, batch_y = next(test_base)
    test_images = batch_x
    test_labels = batch_y  # Get the true labels

    # Make predictions
    predictions = model.predict(test_images)
    main_predictions = predictions['main_predictions']

    fig, axes = plt.subplots(num_samples, 3, figsize=(15, num_samples * 5))
    if num_samples == 1:
        axes = axes.reshape(1, -1)

    # Function to get gradients and create CAM
    def get_grad_cam(input_model, img_array, last_conv_layer_name, pred_index=None):
        # Create a model that maps the input image to the activations of the last conv layer
        # and the model's final predictions
        grad_model = Model(
            inputs=[input_model.inputs],
            outputs=[input_model.get_layer(last_conv_layer_name).output, input_model.output['main_predictions']]
        )

        with tf.GradientTape() as tape:
            last_conv_layer_output, preds = grad_model(img_array)
            if pred_index is None:
                pred_index = tf.argmax(preds[0]) # Get the predicted class index
            class_channel = preds[:, pred_index]

            # Compute the gradient of the top predicted class with respect to the last conv layer output
            grads = tape.gradient(class_channel, last_conv_layer_output)

        # This is a vector where each entry is the mean intensity of the gradient
        # over a specific feature map channel
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        # We multiply each channel in the feature map array by "how important this channel is"
        # with respect to the predicted class
        last_conv_layer_output = last_conv_layer_output[0]
        heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)

        # Normalize the heatmap values to be between 0 and 1
        heatmap = tf.maximum(heatmap, 0) / tf.reduce_max(heatmap)
        return heatmap.numpy()

    # Find the name of the last convolutional layer in your backbone
    # Look for conv layers, prioritizing those in residual blocks
    last_conv_layer_name = None
    
    # First, try to find conv layers in residual blocks
    for layer in reversed(model.layers):
        if hasattr(layer, 'layers'):  # Check if it's a composite layer
            for sublayer in reversed(layer.layers):
                if isinstance(sublayer, layers.Conv2D):
                    last_conv_layer_name = sublayer.name
                    break
        elif isinstance(layer, layers.Conv2D) and any(keyword in layer.name.lower() for keyword in ['conv', 'res']):
            last_conv_layer_name = layer.name
            break
        if last_conv_layer_name:
            break
    
    # If no suitable layer found, get any Conv2D layer
    if last_conv_layer_name is None:
        for layer in reversed(model.layers):
            if isinstance(layer, layers.Conv2D):
                last_conv_layer_name = layer.name
                break
    
    if last_conv_layer_name is None:
        print("Warning: No Conv2D layer found for CAM generation. CAM will be skipped.")
        return

    print(f"Using last convolutional layer: {last_conv_layer_name} for CAM.")

    try:
        for i in range(min(num_samples, len(test_images))):
            # Original image
            axes[i, 0].imshow(test_images[i])
            axes[i, 0].set_title(f'Original Image')
            axes[i, 0].axis('off')

            # Predicted class
            pred_class = np.argmax(main_predictions[i])
            true_class = np.argmax(test_labels[i])

            # Generate CAM
            img_array_expanded = np.expand_dims(test_images[i], axis=0)
            try:
                heatmap = get_grad_cam(model, img_array_expanded, last_conv_layer_name, pred_index=pred_class)

                # Overlay heatmap on image
                # Resize heatmap to match image size
                heatmap = cv2.resize(heatmap, (224, 224))  # Use fixed size
                heatmap = np.uint8(255 * heatmap)
                heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

                # Convert image to 0-255 range for overlay
                superimposed_img = heatmap * 0.4 + (test_images[i] * 255).astype(np.uint8)
                superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)

                axes[i, 1].imshow(superimposed_img)
                axes[i, 1].set_title(f'CAM for Predicted Class ({class_names[pred_class]})')
                axes[i, 1].axis('off')
            
            except Exception as e:
                print(f"Error generating CAM for image {i}: {e}")
                axes[i, 1].text(0.5, 0.5, 'CAM\nGeneration\nFailed', 
                               ha='center', va='center', transform=axes[i, 1].transAxes)
                axes[i, 1].set_title(f'CAM Error')
                axes[i, 1].axis('off')

            # Prediction confidence
            confidence = np.max(main_predictions[i])
            axes[i, 2].bar(class_names, main_predictions[i])
            axes[i, 2].set_title(f'Predictions\nTrue: {class_names[true_class]}, Pred: {class_names[pred_class]}\nConf: {confidence:.3f}')
            axes[i, 2].tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.savefig('class_activation_maps.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    except Exception as e:
        print(f"Error in CAM generation: {e}")
        print("Skipping CAM generation and continuing with evaluation...")

def main():
    print("Building MediFusion-Net Enhanced model...")
    
    train_generator, test_generator, train_base, test_base = create_medical_augmentation()
    model = build_medifusion_enhanced()
    
    print("Model architecture summary:")
    model.summary()
    print(f"\nTotal parameters: {model.count_params():,}")
    
    # Enhanced optimizer with gradient clipping and weight decay
    optimizer = Adam(
        learning_rate=1e-4,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-7,
        clipnorm=1.0,  # Gradient clipping
        weight_decay=1e-4  # L2 regularization
    )
    
    model.compile(
        optimizer=optimizer,
        loss={
            'main_predictions': AdaptiveFocalLoss(alpha=0.25, gamma_base=2.0),
            'epistemic_uncertainty': 'mse',
            'aleatoric_uncertainty': 'mse',
            'attention_features': 'mse'
        },
        loss_weights={
            'main_predictions': 1.0,
            'epistemic_uncertainty': 0.1,  # Reduced from 0.2
            'aleatoric_uncertainty': 0.1,  # Reduced from 0.2
            'attention_features': 0.05     # Reduced from 0.1
        },
        metrics={
            'main_predictions': ['accuracy', tf.keras.metrics.AUC(name='auc')]
        }
    )
    
    # Enhanced callbacks with stability checks
    callbacks = [
        EnhancedModelCheckpoint(
            'medifusion_enhanced_best.keras',
            monitor='val_main_predictions_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1,
            save_weights_only=False
        ),
        
        EarlyStopping(
            monitor='val_main_predictions_accuracy',
            patience=10,  # Increased patience
            restore_best_weights=True,
            verbose=1,
            min_delta=0.002,  # More stringent
            mode='max',
            baseline=0.7  # Minimum acceptable accuracy
        ),
        
        ReduceLROnPlateau(
            monitor='val_main_predictions_loss',
            factor=0.5,  # More aggressive reduction
            patience=3,
            min_lr=1e-6,
            verbose=1,
            cooldown=1,
            mode='min',
            min_delta=0.01  # Only reduce if significant improvement
        ),
        
        CSVLogger('medifusion_enhanced_training.csv', append=True),
        UncertaintyMetrics(),
        
        # Additional callback to log gradients (for debugging)
        tf.keras.callbacks.TensorBoard(
            log_dir='./logs',
            histogram_freq=1,
            write_graph=True,
            write_images=True,
            update_freq='epoch'
        )
    ]
    
    steps_per_epoch = math.ceil(train_base.samples / batch_size)
    validation_steps = math.ceil(test_base.samples / batch_size)
    
    print(f"\nTraining configuration:")
    print(f"Steps per epoch: {steps_per_epoch}")
    print(f"Validation steps: {validation_steps}")
    print(f"Classes: {list(train_base.class_indices.keys())}")
    
    # Training Phase 1: Initial training
    print("\n" + "="*50)
    print("PHASE 1: Initial Training")
    print("="*50)
    
    history1 = model.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=initial_epochs,
        validation_data=test_generator,
        validation_steps=validation_steps,
        callbacks=callbacks,
        verbose=1
    )
    
    # Training Phase 2: Fine-tuning
    print("\n" + "="*50)
    print("PHASE 2: Fine-tuning with Reduced Learning Rate")
    print("="*50)
    
    # Recompile with lower learning rate
    model.compile(
        optimizer=Adam(
            learning_rate=1e-5,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-7,
            clipnorm=1.0
        ),
        loss={
            'main_predictions': AdaptiveFocalLoss(alpha=0.25, gamma_base=2.0),
            'epistemic_uncertainty': 'mse',
            'aleatoric_uncertainty': 'mse',
            'attention_features': 'mse'
        },
        loss_weights={
            'main_predictions': 1.0,
            'epistemic_uncertainty': 0.05,  # Further reduced
            'aleatoric_uncertainty': 0.05,  # Further reduced
            'attention_features': 0.02      # Further reduced
        },
        metrics={
            'main_predictions': ['accuracy', tf.keras.metrics.AUC(name='auc')]
        }
    )
    
    history2 = model.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=initial_epochs + fine_tune_epochs,
        initial_epoch=initial_epochs,
        validation_data=test_generator,
        validation_steps=validation_steps,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save final model and evaluation
    model.save('medifusion_enhanced_final.keras')
    with open('medifusion_class_indices.json', 'w') as f:
        json.dump(train_base.class_indices, f)
    
    print("\n" + "="*50)
    print("MODEL EVALUATION")
    print("="*50)

    # Comprehensive evaluation
    test_results = model.evaluate(test_generator, verbose=1, steps=validation_steps) # Specify steps
    print(f"\nTest Results:")
    for name, value in zip(model.metrics_names, test_results):
        print(f"{name}: {value:.4f}")

    # Predictions with uncertainty quantification
    print("\nGenerating predictions with uncertainty estimates...")
    test_base.reset()

    # Collect all predictions and true labels
    all_predictions = {'main_predictions': [], 'epistemic_uncertainty': [],
                      'aleatoric_uncertainty': [], 'attention_features': []}
    all_true_labels = []

    print("Collecting predictions for detailed analysis...")
    for i in range(validation_steps):
        batch_x, batch_y_dict = next(test_generator) # Get dictionary output
        batch_pred = model.predict(batch_x, verbose=0)

        all_predictions['main_predictions'].extend(batch_pred['main_predictions'])
        all_predictions['epistemic_uncertainty'].extend(batch_pred['epistemic_uncertainty'])
        all_predictions['aleatoric_uncertainty'].extend(batch_pred['aleatoric_uncertainty'])
        all_predictions['attention_features'].extend(batch_pred['attention_features'])
        all_true_labels.extend(np.argmax(batch_y_dict['main_predictions'], axis=1)) # Extract main_predictions

    # Convert to numpy arrays
    for key in all_predictions:
        all_predictions[key] = np.array(all_predictions[key])
    all_true_labels = np.array(all_true_labels)

    # Classification metrics
    predicted_labels = np.argmax(all_predictions['main_predictions'], axis=1)
    class_names = list(train_base.class_indices.keys())

    print("\n" + "="*30)
    print("CLASSIFICATION REPORT")
    print("="*30)
    print(classification_report(all_true_labels, predicted_labels,
                              target_names=class_names, digits=4))

    # Confusion Matrix
    cm = confusion_matrix(all_true_labels, predicted_labels)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix - MediFusion Enhanced')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix_enhanced.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Uncertainty Analysis
    print("\n" + "="*30)
    print("UNCERTAINTY ANALYSIS")
    print("="*30)

    epistemic_mean = np.mean(all_predictions['epistemic_uncertainty'])
    aleatoric_mean = np.mean(all_predictions['aleatoric_uncertainty'])
    total_uncertainty = all_predictions['epistemic_uncertainty'] + all_predictions['aleatoric_uncertainty']

    print(f"Mean Epistemic Uncertainty: {epistemic_mean:.4f}")
    print(f"Mean Aleatoric Uncertainty: {aleatoric_mean:.4f}")
    print(f"Mean Total Uncertainty: {np.mean(total_uncertainty):.4f}")

    # Uncertainty vs Accuracy Analysis
    prediction_confidence = np.max(all_predictions['main_predictions'], axis=1)
    correct_predictions = (predicted_labels == all_true_labels).astype(int)

    # Plot uncertainty distributions
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Epistemic uncertainty distribution
    axes[0, 0].hist(all_predictions['epistemic_uncertainty'], bins=30, alpha=0.7, color='blue')
    axes[0, 0].set_title('Epistemic Uncertainty Distribution')
    axes[0, 0].set_xlabel('Epistemic Uncertainty')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].grid(True, alpha=0.3)

    # Aleatoric uncertainty distribution
    axes[0, 1].hist(all_predictions['aleatoric_uncertainty'], bins=30, alpha=0.7, color='red')
    axes[0, 1].set_title('Aleatoric Uncertainty Distribution')
    axes[0, 1].set_xlabel('Aleatoric Uncertainty')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].grid(True, alpha=0.3)

    # Confidence vs Total Uncertainty
    axes[1, 0].scatter(prediction_confidence, total_uncertainty,
                      c=correct_predictions, cmap='RdYlGn', alpha=0.6)
    axes[1, 0].set_title('Confidence vs Total Uncertainty')
    axes[1, 0].set_xlabel('Prediction Confidence')
    axes[1, 0].set_ylabel('Total Uncertainty')
    axes[1, 0].grid(True, alpha=0.3)

    # ROC-like curve for uncertainty
    sorted_indices = np.argsort(total_uncertainty.flatten()) # Flatten for sorting
    sorted_correct = correct_predictions[sorted_indices]
    cumulative_accuracy = np.cumsum(sorted_correct) / np.arange(1, len(sorted_correct) + 1)
    rejection_rate = np.arange(len(sorted_correct)) / len(sorted_correct)

    axes[1, 1].plot(rejection_rate, cumulative_accuracy, 'b-', linewidth=2)
    axes[1, 1].set_title('Accuracy vs Rejection Rate (by Uncertainty)')
    axes[1, 1].set_xlabel('Rejection Rate')
    axes[1, 1].set_ylabel('Accuracy on Remaining Samples')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('uncertainty_analysis_enhanced.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Calibration Analysis
    print("\nPerforming uncertainty calibration analysis...")
    calibration_data = perform_uncertainty_calibration(all_predictions, all_true_labels)
    plot_calibration_curve(calibration_data)

    # Generate Class Activation Maps
    generate_class_activation_maps(model, test_base, class_names, num_samples=5) # Ensure cv2 is installed

    # Training History Visualization (replace the existing plotting section)
    print("\nPlotting training history...")
    
    # Combine histories
    combined_history = {}
    for key in history1.history.keys():
        combined_history[key] = history1.history[key] + history2.history[key]
    
    # Plot training curves
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Check what metrics are available
    available_metrics = list(combined_history.keys())
    print(f"Available metrics: {available_metrics}")
    
    # Main accuracy - check for the actual metric name
    accuracy_key = None
    val_accuracy_key = None
    for key in available_metrics:
        if 'main_predictions_accuracy' in key and 'val' not in key:
            accuracy_key = key
        elif 'val_main_predictions_accuracy' in key:
            val_accuracy_key = key
    
    if accuracy_key and val_accuracy_key:
        axes[0, 0].plot(combined_history[accuracy_key], label='Training Accuracy')
        axes[0, 0].plot(combined_history[val_accuracy_key], label='Validation Accuracy')
        axes[0, 0].set_title('Model Accuracy')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
    else:
        axes[0, 0].text(0.5, 0.5, 'Accuracy\nMetrics\nNot Found', 
                       ha='center', va='center', transform=axes[0, 0].transAxes)
        axes[0, 0].set_title('Model Accuracy (Not Available)')
    
    # Main loss
    if 'main_predictions_loss' in available_metrics and 'val_main_predictions_loss' in available_metrics:
        axes[0, 1].plot(combined_history['main_predictions_loss'], label='Training Loss')
        axes[0, 1].plot(combined_history['val_main_predictions_loss'], label='Validation Loss')
        axes[0, 1].set_title('Model Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
    else:
        axes[0, 1].text(0.5, 0.5, 'Loss\nMetrics\nNot Found', 
                       ha='center', va='center', transform=axes[0, 1].transAxes)
        axes[0, 1].set_title('Model Loss (Not Available)')
    
    # AUC if available
    auc_key = None
    val_auc_key = None
    for key in available_metrics:
        if 'auc' in key.lower() and 'val' not in key:
            auc_key = key
        elif 'val' in key and 'auc' in key.lower():
            val_auc_key = key
    
    if auc_key and val_auc_key:
        axes[0, 2].plot(combined_history[auc_key], label='Training AUC')
        axes[0, 2].plot(combined_history[val_auc_key], label='Validation AUC')
        axes[0, 2].set_title('Model AUC')
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('AUC')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
    else:
        axes[0, 2].text(0.5, 0.5, 'AUC\nMetrics\nNot Found', 
                       ha='center', va='center', transform=axes[0, 2].transAxes)
        axes[0, 2].set_title('Model AUC (Not Available)')
    
    # Total loss
    if 'loss' in available_metrics and 'val_loss' in available_metrics:
        axes[1, 0].plot(combined_history['loss'], label='Training Total Loss')
        axes[1, 0].plot(combined_history['val_loss'], label='Validation Total Loss')
        axes[1, 0].set_title('Total Model Loss')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    else:
        axes[1, 0].text(0.5, 0.5, 'Total Loss\nMetrics\nNot Found', 
                       ha='center', va='center', transform=axes[1, 0].transAxes)
        axes[1, 0].set_title('Total Loss (Not Available)')
    
    # Epistemic uncertainty
    if 'epistemic_uncertainty_loss' in available_metrics and 'val_epistemic_uncertainty_loss' in available_metrics:
        axes[1, 1].plot(combined_history['epistemic_uncertainty_loss'], label='Training Epistemic Loss')
        axes[1, 1].plot(combined_history['val_epistemic_uncertainty_loss'], label='Validation Epistemic Loss')
        axes[1, 1].set_title('Epistemic Uncertainty Loss')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    else:
        axes[1, 1].text(0.5, 0.5, 'Epistemic\nUncertainty\nNot Found', 
                       ha='center', va='center', transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('Epistemic Uncertainty (Not Available)')
    
    # Aleatoric uncertainty
    if 'aleatoric_uncertainty_loss' in available_metrics and 'val_aleatoric_uncertainty_loss' in available_metrics:
        axes[1, 2].plot(combined_history['aleatoric_uncertainty_loss'], label='Training Aleatoric Loss')
        axes[1, 2].plot(combined_history['val_aleatoric_uncertainty_loss'], label='Validation Aleatoric Loss')
        axes[1, 2].set_title('Aleatoric Uncertainty Loss')
        axes[1, 2].set_xlabel('Epoch')
        axes[1, 2].set_ylabel('Loss')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
    else:
        axes[1, 2].text(0.5, 0.5, 'Aleatoric\nUncertainty\nNot Found', 
                       ha='center', va='center', transform=axes[1, 2].transAxes)
        axes[1, 2].set_title('Aleatoric Uncertainty (Not Available)')
    
    plt.tight_layout()
    plt.savefig('training_history_enhanced.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Performance Summary
    print("\n" + "="*50)
    print("FINAL PERFORMANCE SUMMARY")
    print("="*50)

    test_accuracy = np.mean(predicted_labels == all_true_labels)
    high_confidence_mask = prediction_confidence > 0.9
    high_conf_accuracy = np.mean(predicted_labels[high_confidence_mask] == all_true_labels[high_confidence_mask])

    # Calculate per-class metrics
    per_class_metrics = {}
    for i, class_name in enumerate(class_names):
        class_mask = all_true_labels == i
        if np.sum(class_mask) > 0:
            class_accuracy = np.mean(predicted_labels[class_mask] == all_true_labels[class_mask])
            class_confidence = np.mean(prediction_confidence[class_mask])
            class_uncertainty = np.mean(total_uncertainty[class_mask])

            per_class_metrics[class_name] = {
                'accuracy': class_accuracy,
                'avg_confidence': class_confidence,
                'avg_uncertainty': class_uncertainty,
                'sample_count': np.sum(class_mask)
            }

    print(f"Overall Test Accuracy: {test_accuracy:.4f}")
    print(f"High Confidence Accuracy (>0.9): {high_conf_accuracy:.4f}")
    print(f"High Confidence Samples: {np.sum(high_confidence_mask)}/{len(prediction_confidence)} ({np.sum(high_confidence_mask)/len(prediction_confidence)*100:.1f}%)")
    print(f"Mean Prediction Confidence: {np.mean(prediction_confidence):.4f}")
    print(f"Mean Total Uncertainty: {np.mean(total_uncertainty):.4f}")

    print("\nPer-Class Performance:")
    print("-" * 60)
    for class_name, metrics in per_class_metrics.items():
        print(f"{class_name:15} | Acc: {metrics['accuracy']:.4f} | "
              f"Conf: {metrics['avg_confidence']:.4f} | "
              f"Unc: {metrics['avg_uncertainty']:.4f} | "
              f"Samples: {metrics['sample_count']:3d}")

    # Save comprehensive results
    results_summary = {
        'test_accuracy': float(test_accuracy),
        'high_confidence_accuracy': float(high_conf_accuracy),
        'mean_confidence': float(np.mean(prediction_confidence)),
        'mean_epistemic_uncertainty': float(epistemic_mean),
        'mean_aleatoric_uncertainty': float(aleatoric_mean),
        'mean_total_uncertainty': float(np.mean(total_uncertainty)),
        'per_class_metrics': {k: {mk: float(mv) if isinstance(mv, (np.floating, float)) else int(mv)
                             for mk, mv in v.items()} for k, v in per_class_metrics.items()},
        'model_params': int(model.count_params()),
        'training_epochs': initial_epochs + fine_tune_epochs,
        'class_names': class_names
    }

    with open('medifusion_enhanced_results.json', 'w') as f:
        json.dump(results_summary, f, indent=2)

    print(f"\nDetailed results saved to 'medifusion_enhanced_results.json'")
    print(f"Model saved as 'medifusion_enhanced_final.keras'") # Changed to .keras
    print(f"Class indices saved as 'medifusion_class_indices.json'")

    # Final model insights
    print("\n" + "="*50)
    print("MODEL INSIGHTS & RECOMMENDATIONS")
    print("="*50)

    if test_accuracy > 0.9:
        print("✅ Excellent performance achieved!")
    elif test_accuracy > 0.8:
        print("✅ Good performance achieved!")
    else:
        print("⚠️  Consider additional training or data augmentation")

    if np.mean(total_uncertainty) < 0.3:
        print("✅ Model shows good confidence in predictions")
    else:
        print("⚠️  High uncertainty detected - consider more training data")

    print(f"\nKey Features of MediFusion-Net Enhanced:")
    print(f"• Vision Transformer hybrid architecture")
    print(f"• Adaptive Multi-Scale Attention (AMSA)")
    print(f"• Bayesian uncertainty quantification")
    print(f"• Graph-inspired anatomical reasoning")
    print(f"• Advanced focal loss for medical imbalance")
    print(f"• Squeeze-and-Excitation attention blocks")
    print(f"• Comprehensive interpretability features")

    print(f"\nTotal training time: {initial_epochs + fine_tune_epochs} epochs")
    print(f"Model parameters: {model.count_params():,}")
    print("Training completed successfully! 🎉")

if __name__ == "__main__":
    main()