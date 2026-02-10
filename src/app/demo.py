"""Gradio demo application for medical image augmentation."""

import gradio as gr
import torch
from pathlib import Path
import sys
from PIL import Image
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.inference import ImageGenerator, save_generation_result
from src.data import DR_GRADE_TO_TEXT


class MedicalImageAugmentationDemo:
    """Gradio demo for medical image augmentation system."""
    
    def __init__(self, checkpoint_path: str):
        """
        Initialize demo.
        
        Args:
            checkpoint_path: Path to trained model checkpoint
        """
        self.checkpoint_path = checkpoint_path
        self.generator = None
        
        # Check if checkpoint exists
        if not Path(checkpoint_path).exists():
            print(f"Warning: Checkpoint not found: {checkpoint_path}")
            print("Demo will run in mock mode (no actual generation)")
            self.mock_mode = True
        else:
            self.mock_mode = False
            self._load_generator()
    
    def _load_generator(self):
        """Load the image generator."""
        try:
            print("Loading model...")
            self.generator = ImageGenerator(
                checkpoint_path=self.checkpoint_path,
                device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            )
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {e}")
            self.mock_mode = True
    
    def generate_image(
        self,
        reference_image: np.ndarray,
        mask_image: np.ndarray,
        text_description: str,
        dr_grade: str,
        num_steps: int,
        guidance_scale: float,
        seed: int,
        use_mask: bool
    ):
        """
        Generate image based on inputs.
        
        Args:
            reference_image: Reference fundus image (optional)
            mask_image: Segmentation mask
            text_description: Text description
            dr_grade: DR grade selection
            num_steps: Number of sampling steps
            guidance_scale: CFG scale
            seed: Random seed
            use_mask: Whether to use mask conditioning
            
        Returns:
            Tuple of (generated_image, status_message)
        """
        try:
            # Validate inputs
            if mask_image is None and not text_description and dr_grade == "None":
                return None, "❌ Error: Please provide at least a mask or text description"
            
            # Prepare caption
            if dr_grade != "None":
                caption = DR_GRADE_TO_TEXT[int(dr_grade)]
            elif text_description:
                caption = text_description
            else:
                caption = "Diabetic retinopathy fundus image"
            
            # Prepare mask
            mask = None
            if use_mask and mask_image is not None:
                mask = Image.fromarray(mask_image)
            
            # Mock mode for testing without model
            if self.mock_mode:
                # Create a dummy image
                dummy_image = Image.new('RGB', (512, 512), color=(100, 150, 200))
                status = f"✓ Mock generation completed\n"
                status += f"Caption: {caption}\n"
                status += f"Mask: {'Yes' if mask else 'No'}\n"
                status += f"Steps: {num_steps}, CFG: {guidance_scale}, Seed: {seed}"
                return dummy_image, status
            
            # Real generation
            print(f"Generating image with caption: {caption}")
            generated_image = self.generator.generate(
                caption=caption,
                mask=mask,
                image_size=1024,
                num_inference_steps=num_steps,
                guidance_scale=guidance_scale,
                seed=seed if seed >= 0 else None,
                sampler_type="ddim"
            )
            
            # Save result
            metadata = {
                'caption': caption,
                'dr_grade': dr_grade if dr_grade != "None" else None,
                'num_steps': num_steps,
                'guidance_scale': guidance_scale,
                'seed': seed,
                'use_mask': use_mask
            }
            
            save_paths = save_generation_result(
                generated_image,
                output_dir="results",
                metadata=metadata
            )
            
            status = f"✓ Generation completed!\n"
            status += f"Caption: {caption}\n"
            status += f"Saved to: {save_paths['result_dir']}"
            
            return generated_image, status
            
        except Exception as e:
            import traceback
            error_msg = f"❌ Error during generation:\n{str(e)}\n\n{traceback.format_exc()}"
            return None, error_msg
    
    def create_interface(self):
        """Create Gradio interface."""
        
        with gr.Blocks(title="Medical Image Augmentation System") as demo:
            gr.Markdown("""
            # 🏥 Medical Image Augmentation System
            
            Generate synthetic diabetic retinopathy fundus images using text and segmentation mask conditioning.
            
            **Features:**
            - Text-to-image generation with DR grade selection
            - Mask-conditioned generation for precise lesion control
            - Configurable sampling parameters
            """)
            
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 📥 Input")
                    
                    # Reference image (optional)
                    reference_img = gr.Image(
                        label="Reference Image (Optional)",
                        type="numpy",
                        sources=["upload"],
                        height=256
                    )
                    
                    # Mask input
                    mask_img = gr.Image(
                        label="Segmentation Mask",
                        type="numpy",
                        sources=["upload"],
                        height=256
                    )
                    
                    use_mask = gr.Checkbox(
                        label="Use mask conditioning",
                        value=True
                    )
                    
                    # Text inputs
                    gr.Markdown("### 📝 Text Description")
                    
                    dr_grade = gr.Radio(
                        choices=["None", "0", "1", "2", "3", "4"],
                        value="None",
                        label="DR Grade (overrides text description)",
                        info="0=No DR, 1=Mild, 2=Moderate, 3=Severe, 4=Proliferative"
                    )
                    
                    text_desc = gr.Textbox(
                        label="Custom Text Description",
                        placeholder="E.g., Fundus image with microaneurysms and hemorrhages",
                        lines=3
                    )
                    
                    # Generation parameters
                    gr.Markdown("### ⚙️ Generation Parameters")
                    
                    num_steps = gr.Slider(
                        minimum=10,
                        maximum=100,
                        value=50,
                        step=5,
                        label="Sampling Steps",
                        info="More steps = better quality but slower"
                    )
                    
                    guidance_scale = gr.Slider(
                        minimum=1.0,
                        maximum=15.0,
                        value=7.5,
                        step=0.5,
                        label="Guidance Scale",
                        info="Higher = more faithful to text"
                    )
                    
                    seed = gr.Number(
                        value=-1,
                        label="Random Seed",
                        info="-1 for random"
                    )
                    
                    generate_btn = gr.Button(
                        "🎨 Generate Image",
                        variant="primary",
                        size="lg"
                    )
                
                with gr.Column(scale=1):
                    gr.Markdown("### 📤 Output")
                    
                    output_img = gr.Image(
                        label="Generated Image",
                        type="pil",
                        height=512
                    )
                    
                    status_text = gr.Textbox(
                        label="Status",
                        lines=5,
                        max_lines=10
                    )
            
            # Examples
            gr.Markdown("### 📚 Examples")
            gr.Examples(
                examples=[
                    [None, None, "Healthy fundus image", "0", 50, 7.5, 42, False],
                    [None, None, "Mild diabetic retinopathy", "1", 50, 7.5, 123, False],
                    [None, None, "Severe diabetic retinopathy with hemorrhages", "3", 50, 7.5, 456, False],
                ],
                inputs=[
                    reference_img,
                    mask_img,
                    text_desc,
                    dr_grade,
                    num_steps,
                    guidance_scale,
                    seed,
                    use_mask
                ],
                label="Try these examples"
            )
            
            # Event handlers
            generate_btn.click(
                fn=self.generate_image,
                inputs=[
                    reference_img,
                    mask_img,
                    text_desc,
                    dr_grade,
                    num_steps,
                    guidance_scale,
                    seed,
                    use_mask
                ],
                outputs=[output_img, status_text]
            )
            
            gr.Markdown("""
            ---
            ### 📖 Instructions
            
            1. **Upload a segmentation mask** (optional) - Binary mask highlighting lesion regions
            2. **Select DR grade** or enter custom text description
            3. **Adjust generation parameters** as needed
            4. **Click Generate** to create synthetic fundus image
            
            ### 💡 Tips
            
            - Use mask conditioning for precise control over lesion locations
            - Higher sampling steps (50-100) produce better quality
            - Guidance scale 7-10 works well for medical images
            - Set a seed for reproducible results
            
            ### ⚠️ Note
            
            This is a research prototype. Generated images should not be used for clinical diagnosis.
            """)
        
        return demo


def main():
    """Main function to launch demo."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Launch Gradio demo")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/best_model.pth",
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create public share link"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Port to run on"
    )
    args = parser.parse_args()
    
    # Create demo
    demo_app = MedicalImageAugmentationDemo(args.checkpoint)
    demo = demo_app.create_interface()
    
    # Launch
    print("\n" + "="*70)
    print("LAUNCHING GRADIO DEMO")
    print("="*70)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Port: {args.port}")
    print(f"Share: {args.share}")
    print("="*70 + "\n")
    
    demo.launch(
        server_name="0.0.0.0",
        server_port=args.port,
        share=args.share
    )


if __name__ == "__main__":
    main()
