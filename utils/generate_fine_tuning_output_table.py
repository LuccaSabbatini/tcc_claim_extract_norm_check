import os
import pandas as pd

DATASETS_PATH = "../data"
LATEX_OUTPUT_FILE = "../assets/LaTeX/apendex-fine-tuning-results.tex"
TRANSLATIONS = {
    "claim_extraction": "\\textit{Claim Extraction}",
    "claim_normalization": "\\textit{Claim Normalization}",
    "original": "Original",
    "gpt-5": "GPT-5",
    "gpt-5-nano": "GPT-5 nano",
    "xlm-roberta-large": "XLM-RoBERTa-Large",
    "bert-large-portuguese-cased": "BERTimbau",
    "fakebr": "Fake.br",
    "faketweetbr": "FakeTweet.BR",
    "fakerecogna": "FakeRecogna",
}

file_output_lines = []


# Function to generate LaTeX tables for executions
def generate_executions_tables(version_path, transformer, dataset, version):
    executions = os.listdir(version_path)
    executions_counter = 0
    all_executions_metrics = pd.DataFrame()

    for execution in executions:
        executions_counter += 1
        execution_path = f"{version_path}/{execution}"
        metrics_file = f"{execution_path}/metrics/epoch_metrics.csv"

        if os.path.exists(metrics_file):
            df = pd.read_csv(metrics_file)
            best_epoch = df.loc[df["eval_loss"].idxmin()]

            execution_metrics = {
                "Treino": f"Treino {executions_counter}",
                "Época do melhor modelo": best_epoch["epoch"].astype(int),
                "Perda no treinamento": best_epoch["training_loss"].astype(float),
                "Perda na validação": best_epoch["eval_loss"].astype(float),
                "Acurácia": best_epoch["accuracy"].astype(float),
                "Precisão": best_epoch["precision"].astype(float),
                "F1": best_epoch["f1"].astype(float),
                "Recall": best_epoch["recall"].astype(float),
            }

            # Append execution metrics to DataFrame
            all_executions_metrics = pd.concat(
                [
                    all_executions_metrics,
                    pd.DataFrame([execution_metrics]),
                ],
                ignore_index=True,
            )

    # Calculate averages across executions and add it to column "Média"
    executions_average = {
        "Treino": "Média",
        "Época do melhor modelo": all_executions_metrics[
            "Época do melhor modelo"
        ].mean(),
        "Perda no treinamento": all_executions_metrics["Perda no treinamento"]
        .mean()
        .astype(float),
        "Perda na validação": all_executions_metrics["Perda na validação"]
        .mean()
        .astype(float),
        "Acurácia": all_executions_metrics["Acurácia"].mean().astype(float),
        "Precisão": all_executions_metrics["Precisão"].mean().astype(float),
        "F1": all_executions_metrics["F1"].mean().astype(float),
        "Recall": all_executions_metrics["Recall"].mean().astype(float),
    }

    all_executions_metrics = pd.concat(
        [
            all_executions_metrics,
            pd.DataFrame([executions_average]),
        ],
        ignore_index=True,
    )

    # Generate lines for LaTeX table
    lines = []

    for metric in [
        "Época do melhor modelo",
        "Perda no treinamento",
        "Perda na validação",
        "Acurácia",
        "Precisão",
        "F1",
        "Recall",
    ]:
        line = f"{metric} & "
        for i in range(executions_counter):
            if metric == "Época do melhor modelo":
                line += f"{int(all_executions_metrics.iloc[i][metric])} & "
            else:
                line += f"{all_executions_metrics.iloc[i][metric]:.3f} & "
        line += f"{float(all_executions_metrics.loc[len(all_executions_metrics) - 1][metric]):.3f} \\\\ \n"
        lines.append(line)
        continue

    # Generate LaTeX table
    file_output_lines.append(f"% {transformer} - {version} - {dataset} \n")
    file_output_lines.append("\\begin{table}[!htbp]\n")
    file_output_lines.append("\\centering\n")
    file_output_lines.append("\\begin{tabular}{l*4c}\n")
    file_output_lines.append("\\toprule\n")
    file_output_lines.append(
        f"\\multicolumn{{5}}{{c}}{{\\textbf{{{TRANSLATIONS[transformer]} | {TRANSLATIONS[dataset]} | {version}}}}} \\\\ \n"
    )
    file_output_lines.append("\\midrule\n")
    file_output_lines.append(
        "\\textbf{Métrica} & \\textbf{Treino 1} & \\textbf{Treino 2} & \\textbf{Treino 3} & \\textbf{Média} \\\\ \n"
    )
    file_output_lines.append("\\midrule\n")

    for line in lines:
        file_output_lines.append(line)

    file_output_lines.append("\\bottomrule\n")
    file_output_lines.append("\\end{tabular}\n")
    file_output_lines.append(
        "\\caption{"
        + f"Métricas dos melhores modelos de cada ajuste-fino do \\textit{{Transformer}} {TRANSLATIONS[transformer]} treinando na base {TRANSLATIONS[dataset]} {version}."
        + "}\n"
    )
    file_output_lines.append("\\end{table}\n\n")


# Main loop to process datasets, transformers, and versions
def main():
    datasets = os.listdir(DATASETS_PATH)

    for dataset in datasets:
        results_path = f"{DATASETS_PATH}/{dataset}/results"
        transformers = os.listdir(results_path) if os.path.exists(results_path) else []

        for transformer in transformers:
            transformer_path = f"{results_path}/{transformer}"
            dataset_versions = os.listdir(transformer_path)

            for version in dataset_versions:
                version_path = f"{transformer_path}/{version}"

                if version == "original":
                    generate_executions_tables(
                        version_path, transformer, dataset, TRANSLATIONS[version]
                    )
                else:
                    llms = os.listdir(version_path)

                    for llm in llms:
                        llm_path = f"{version_path}/{llm}"
                        generate_executions_tables(
                            llm_path,
                            transformer,
                            dataset,
                            f"{TRANSLATIONS[version]} ({TRANSLATIONS[llm.split('_')[0]]})",
                        )


main()

# Write output to LaTeX file
with open(LATEX_OUTPUT_FILE, "w", encoding="utf-8") as latex_file:
    latex_file.writelines(file_output_lines)
