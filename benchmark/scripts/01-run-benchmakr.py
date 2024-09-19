import logging
from pathlib import Path
import torch
import shutil
import traceback

# Example usage
logging.basicConfig(level=logging.INFO, force=True)
torch.manual_seed(0)

repo_dir = Path(__file__).parent.parent

resources_dir = repo_dir / "resources"
eval_dir = repo_dir / "eval"


def query_cad_object(cad_file: Path, question: str) -> str:
    """this method should call your approach for CAD question answering

    you can save all debugging information in $(cwd)/out folder. This folder
    will be moved to the sample folder in the evaluation folder after each sample

    :param cad_file: file path to the cad (.step) file to evaluate
    :param question: question about `cad_file`
    :return: answer computed by your CAD question answering approach
    """
    print("Evaluate %s" % cad_file.as_posix())
    # load cad_file and answer question `question`
    return None


for ds_dir in sorted(resources_dir.iterdir()):
    if not ds_dir.is_dir():
        continue

    print("=" * 100)
    print("Dataset:", ds_dir.name)
    print("=" * 100)
    for sample in sorted(ds_dir.iterdir()):
        # sample_name = "00000024"
        # sample = abc_out_dir / sample_name
        eval_sample_dir = eval_dir / ds_dir.name / sample.name
        eval_sample_dir.mkdir(exist_ok=True, parents=True)
        print("-" * 100)
        print("Sample: %s - %s" % (ds_dir.name, sample.name))
        print("-" * 100)

        cad_file = sample / "cad.step"

        with open(sample / "input.txt") as f:
            text_batch = f.read()

        text_batch = text_batch.split("\n")

        for i, question in enumerate(text_batch):
            eval_sample_text_dir = eval_sample_dir / str(i)

            if eval_sample_text_dir.exists():
                if (eval_sample_text_dir / "result.txt").is_file():
                    with open(eval_sample_text_dir / "result.txt") as f:
                        data = f.read()
                    if data == "None":
                        print("Data is none: Deleting")
                        shutil.rmtree(eval_sample_text_dir)
                    else:
                        continue
                else:
                    if (eval_sample_text_dir / "exception.txt").is_file():
                        # retry on exception
                        # shutil.rmtree(eval_sample_text_dir)
                        # print("%d: %s, retry on exception" % (i, text))
                        continue
                    else:
                        print("%d: %s, already answered" % (i, question))
                        continue

            print("%d: %s" % (i, question))

            out_dir = Path("./out")
            if out_dir.is_dir():
                shutil.rmtree("./out")
            out_dir.mkdir()

            try:
                # load cad_file and answer question `question`
                response = query_cad_object(cad_file=cad_file, question=question)

                with open("./out/result.txt", "w") as f:
                    f.write(str(response))

                print(response)
            except Exception as e:
                traceback.print_exc()
                with open("./out/exception.txt", "w") as f:
                    f.write(str(e))
                    f.write(traceback.format_exc())

            shutil.move(out_dir, eval_sample_text_dir)
