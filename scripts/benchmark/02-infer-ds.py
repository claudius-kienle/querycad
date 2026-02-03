import logging
import shutil
import traceback
from pathlib import Path

import torch
from dotenv import load_dotenv
from llm_utils.textgen_api.textgen_api import TextGenApi

from cad_service.cad_expert_llm_interface import CADExpertLLMInterface
from cad_service.cad_expert_python_interpreter import CADExpertPythonInterpreter
from cad_service.img_seg.image_cad_segmentation_esemble import ImageCADSegmentationEnsemble
from cad_service.img_seg.image_processing_client import ImageProcessingClient
from cad_service.kernel.cad_loader import CADLoader

# Example usage
logging.basicConfig(level=logging.INFO, force=True)
torch.manual_seed(0)
load_dotenv()

root_dir = Path(__file__).parent.parent.parent
resources_dir = root_dir / "resources"
dss_dir = resources_dir / "cad-models"
eval_dir = resources_dir / "eval"


checkpoint_path = (
    root_dir
    / "resources/model-checkpoints/gnn/MF_CAD++_residual_lvl_7_edge_MFCAD++_units_512_date_2021-07-27_epochs_100.ckpt"
)
loader = CADLoader()
out_dir = Path("./out")
client = ImageProcessingClient(api_url="http://localhost:8000")
img_seg = ImageCADSegmentationEnsemble(
    visualize=True,
    image_processing_client=client,
    image_size=(1920, 1080),
    ray_casting_downscale_factor=20,
    segmentation_threshold=0.3,
)

llm = TextGenApi.default("gpt4o-mini")
cad_expert_llm = CADExpertLLMInterface(llm_service=llm)
viz_segmentation = True
merge_shapes = True


for ds_dir in sorted(dss_dir.iterdir()):
    if not ds_dir.is_dir():
        continue

    print("=" * 100)
    print("Dataset:", ds_dir.name)
    print("=" * 100)
    ds_processed_dir = ds_dir / "processed"
    for sample in sorted(ds_processed_dir.iterdir()):
        eval_sample_dir = eval_dir / ds_dir.name / "processed" / sample.name
        eval_sample_dir.mkdir(exist_ok=True, parents=True)
        print("-" * 100)
        print("Sample: %s - %s" % (ds_dir.name, sample.name))
        print("-" * 100)

        cad_file = sample / "cad.step"

        with open(sample / "input.txt") as f:
            text_batch = f.read()

        text_batch = text_batch.split("\n")

        for i, text in enumerate(text_batch):
            if out_dir.is_dir():
                shutil.rmtree("./out")

            out_dir.mkdir()
            eval_sample_text_dir = eval_sample_dir / str(i)

            if text.startswith("X "):
                print(text)
                continue

            if eval_sample_text_dir.exists():
                if (eval_sample_text_dir / "result.txt").is_file():
                    with open(eval_sample_text_dir / "result.txt") as f:
                        data = f.read()
                    print("tmp%stmp" % data)
                    if data == "None":
                        print("Data is none: Deleting")
                        shutil.rmtree(eval_sample_text_dir)
                    else:
                        continue
                else:
                    if (eval_sample_text_dir / "exception.txt").is_file():
                        # retry on exception
                        shutil.rmtree(eval_sample_text_dir)
                        print("%d: %s, retry on exception" % (i, text))
                        # continue
                    else:
                        print("%d: %s, already answered" % (i, text))
                        continue

            print("%d: %s" % (i, text))

            shape = loader.read_shape(step_file_path=cad_file)
            if merge_shapes:
                shape = loader.fuse_shape_subshapes(shape)
            interpreter = CADExpertPythonInterpreter(shape=shape, image_seg=img_seg, visualize=viz_segmentation)

            try:
                code_sections = cad_expert_llm.query_cad_object(query=text, log=True)
                assert code_sections is not None, "No code sections returned from LLM"

                response = interpreter.exec(code_sections)

                with open("./out/result.txt", "w") as f:
                    f.write(str(response))

                print(response)
            except Exception as e:
                traceback.print_exc()
                with open("./out/exception.txt", "w") as f:
                    f.write(str(e))
                    f.write(traceback.format_exc())

            # if eval_sample_text_dir.exists():
            #     shutil.rmtree(eval_sample_text_dir)
            shutil.move(out_dir, eval_sample_text_dir)
            # eval_sample_text_dir.mkdir(exist_ok=True)
