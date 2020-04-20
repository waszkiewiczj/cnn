import os
import tester.plots


class TestResultsSaver:
    def __init__(self, test_name, output_dir_path):
        self.test_name = test_name
        self.test_dir_path = f"{output_dir_path}/{test_name}"
        os.makedirs(self.test_dir_path, exist_ok=True)

    def save_partial_results(self, test_it, df):
        partial_result_filepath = f"{self.test_dir_path}/raw_results_{test_it}.csv"
        df.to_csv(partial_result_filepath, index=False)

    def save_full_results(self, df):
        raw_results_filepath = f"{self.test_dir_path}/raw_results.csv"
        results_filepath = f"{self.test_dir_path}/results.csv"
        accuracy_plot_filepath = f"{self.test_dir_path}/accuracy.svg"
        loss_plot_filepath = f"{self.test_dir_path}/loss.svg"
        df.to_csv(raw_results_filepath, index=False)
        grouped_df = self.__get_grouped_results(df)
        grouped_df.to_csv(results_filepath, index=False)
        tester.plots.create_accuracy_plot(grouped_df).savefig(accuracy_plot_filepath)
        tester.plots.create_loss_plot(grouped_df).savefig(loss_plot_filepath)

    @staticmethod
    def __get_grouped_results(df):
        grouped = df.groupby('epoch')
        means = grouped.mean().add_prefix('mean_')
        stds = grouped.std().add_prefix('std_')
        return means.join(stds).reset_index()
