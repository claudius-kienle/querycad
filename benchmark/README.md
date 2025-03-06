# QueryCAD Benchmark

## Getting Started

Install the virtual environment with [pixi](https://pixi.sh)
```bash
pixi install
```

Unzip the [./resources/cad-benchmark.zip](./resources/cad-benchmark.zip)
```bash
cd resources
unzip cad-benchmark.zip
# to download the missing cad files, see ./resources/README.md
cd ..
```

To run the benchmark on your approach, execute
```bash
cd scripts
python 01-run-benchmark.py
```

Adapt the `query_cad_object` function to call you approach for CAD question answering.
This script will create a `eval` folder at the repo root with the evaluation results of your approach.

Finally, you can evaluate your approach with the [./scripts/02-evaluate-approach.ipynb](./scripts/02-evaluate-approach.ipynb) jupyter notebook.

To compare evaluations, use [./scripts/03-compare-eval.ipynb](./scripts/03-compare-eval.ipynb).


## Details about the Benchmark

This is a benchmark for CAD question answering. 

It consists of 18 CAD models, 10 of which are derived from the ABC dataset, located at [./resources/abc](./resources/abc), and 8 CAD models stem from real-world industrial settings, located at [./resources/industrial](./resources/industrial).

For each CAD model, we generated between 4 and 10 questions stored in a file `input.txt`. They are designed to retrieve specific properties, such as measurements, positions, or counts of particular parts. Each question targets either a specific type of part or the entire CAD model, inquiring about a particular property while sometimes restricting the valid parts by specifying a side the parts should be visible on or applying other filtering criteria based on the part's characteristics. The distribution and structure of these questions are illustrated in following Figure:
![Dataset Overview](./assets/ds-overview.svg)

To address open-set segmentation, the questions cover a total of 43 diverse parts and ask about 11 different properties.

To define the label for each question, we manually measured the CAD models using a CAD kernel and calculated the responses by hand, saved in a `solution.txt` file.

Sometimes, there are multiple correct answers to a question. In such cases, `solution.txt` contains a regular expression (prefixed with `re`) the answer must respect. Also, if the answer is a unsorted list, we sort it before comparison, toggled via the `sorted` keyword in `metadata.txt`.