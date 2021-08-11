import os
import cog
import subprocess
import tempfile
from pathlib import Path


class NeuralStylePredictor(cog.Predictor):
    def setup(self):
        """setup"""
        pass
    @cog.input("input_content", type=Path, help="Content image path")
    @cog.input("input_style", type=Path, help="Style image path")
    @cog.input("model_path", type=str, help="Pre-trained model path",
               default='imagenet-vgg-verydeep-19.mat')
    @cog.input("blend", type=float, help="Coefficient of content transfer layers (0.0/1.0)",
                default=1.0)


    def predict(self, input_content, input_style, model_path, blend):
        """Compute neural style transfer between 2 images"""
        #compute prediction
        output_path = Path(tempfile.mkdtemp()) / "output.jpg"
        args = ['python ', 'neural_style.py ']
        args.append(' --content ' + str(input_content))
        args.append(' --styles ' + str(input_style))
        args.append(' --output ' + str(output_path))
        args.append(' --network ' + model_path)
        args.append(' --content-weight-blend ' + str(blend))
        p = subprocess.Popen("".join(args), shell=True)
        p.communicate()

        return output_path
