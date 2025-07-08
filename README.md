This repository contains the code, datasets, and graphs created and developed for my bachelor's thesis project: <b>Quickfix, A multistep children online search query reformulation method using LLM</b>
The paper is available [here](https://resolver.tudelft.nl/uuid:8a8315a1-cccc-4115-baa1-bf8633983f40).


# Repository Description 
Below are the three main folders in our repository and their description. 

### datasets
This folder contains the initial dataset, as well as intermediary and final datasets, for the experiments. 
* `ChildrenQueries.csv` is the Children English Queries dataset used in the evaluation of our method. It's also available at: https://scholarworks.boisestate.edu/cs_scripts/5/
* `reformulated_queries_debug.csv` is the reformulated version of the original `ChildrenQueries`. For each original query, it contains the intermediate steps of the complete LLM reformulation, as well as individual rules ablation reformulations.
* `query_metrics.csv` is the final scores calculated version of our reformulated queries. It contains the following attributes: `query id`, `query variant`, `snippets (retrieved)`, `snippets count (for safety checks)`, and an attribute for each of the 7 scores used in our evaluation (3 readability scores and 4 content safety scores calculated using Perspective API).

### output
This folder contains graphs produced using our final analysis dataset, showcasing the demonstrations and performance of our proposed method. 

### scripts
This folder contains the code used for our proposed method as well as analysis codes. 
* `reformulation.py` contains the actual script and the code for our method, used to reformulate each original query.
* `evaluation.py` contains the script used to accumulate web results for each query, as well as the evaluation scores calculation for them. While running this, you also need to create the output csv file beforehand with the proper headers, as shown in the respective output csv in the `datasets` folder.
* `statistical_analysis.py` contains the statistical analysis code carried out as part of our experiments. Considering `alpha = 0.05`, this also contains appropriate statistical significance tests for analysing whether the impact of a reformulation is significant. It also produces descriptive statistics such as the median difference of the reformulated query scores vs. the original query counterparts. It also contains the graph code used for the ones in the `output` folder.

---
# How-to Run 
1) You need the proper api keys in order. An example of what's needed is shown in `.env.example`. Acquire the keys, fill the example, and move it to an actual `.env` file. You can request the `Perspective API Key` [here](https://developers.perspectiveapi.com/s/docs-enable-the-api?language=en_US), `Brave API Key` [here](https://api-dashboard.search.brave.com/register), and `Gemini API Key` [here](https://ai.google.dev/gemini-api/docs/api-key).
2) Then, you need to install the necessary packages by running the following from the project root (<b>Optional:</b> `create` and `activate` a virtual environment before you install the packages): `pip install -r requirements.txt`.
3) Move the appropriate dataset from `datasets` and the script from `scripts` to the same folder before you run it. Or you can adjust the path used in the script file to match the `datasets` folder directly. 
---

# Contact 
If you have further questions regarding the repository or my thesis, reach out to me at atilla[dot]colak[at]outlook[dot]com.
