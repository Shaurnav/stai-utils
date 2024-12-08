import torch.nn as nn


class FinetuneModel(nn.Module):
    def __init__(
        self, encoder, output_dim, dim=3, use_checkpoint=False, pred_head_type="mlp"
    ):
        """KeyMorph pipeline in a single module. Used for training.

        :param backbone: Backbone network
        :param num_keypoints: Number of keypoints
        :param dim: Dimension
        :param keypoint_extractor: Keypoint extractor
        :param max_train_keypoints: Maximum number of keypoints to use during training
        """
        super().__init__()
        self.encoder = encoder
        self.output_dim = output_dim
        self.dim = dim
        self.use_checkpoint = use_checkpoint

        if pred_head_type == "linear":
            self.pred_head = nn.Sequential(
                nn.AdaptiveAvgPool3d((1, 1, 1)),
                nn.Flatten(),
                nn.Linear(256, output_dim),
            )
        elif pred_head_type == "mlp":
            self.pred_head = nn.Sequential(
                nn.AdaptiveAvgPool3d((1, 1, 1)),
                nn.Flatten(),
                nn.Linear(256, 512),
                nn.ReLU(),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, output_dim),
            )
        else:
            raise ValueError(f"Invalid pred_head_type: {pred_head_type}")

    def forward(self, img, return_feat=False):
        """Forward pass for one mini-batch step.

        :param img_f, img_m: Fixed and moving images
        :param transform_type: str or tuple of str of keypoint alignment types. Used for finding registrations
            for multiple alignment types in one forward pass, without having to extract keypoints
            every time.

        :return res: Dictionary of results
        """
        assert img.shape[1] == 1, "Image dimension must be 1"

        result_dict = {}

        encoder_output = self.encoder(img)
        pred_output = self.pred_head(encoder_output)
        result_dict = {"enc_out": encoder_output, "pred_out": pred_output}

        if return_feat:
            result_dict["feat"] = encoder_output

        return result_dict
