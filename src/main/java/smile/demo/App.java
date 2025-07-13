package smile.demo;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.List;

import smile.classification.LogisticRegression;

public class App {
    public static void main(String[] args) throws Exception {
        String filePath = "game_data.csv";

        List<double[]> features = new ArrayList<>();
        List<Integer> labels = new ArrayList<>();

        try (BufferedReader br = new BufferedReader(new FileReader(filePath))) {
            String line = br.readLine(); 
            while ((line = br.readLine()) != null) {
                String[] parts = line.split(",");
                double[] row = new double[parts.length - 1];
                for (int i = 0; i < row.length; i++) {
                    row[i] = Double.parseDouble(parts[i]);
                }
                features.add(row);
                labels.add(Integer.parseInt(parts[parts.length - 1]));
            }
        }

        double[][] X = features.toArray(new double[0][]);
        int[] y = labels.stream().mapToInt(i -> i).toArray();

        LogisticRegression.Binomial model = LogisticRegression.binomial(X, y);

        System.out.println("Predicted probabilities of winning:");
        for (int i = 0; i < X.length; i++) {
            double linearPredictor = model.score(X[i]);
            double prob = 1.0 / (1.0 + Math.exp(-linearPredictor));
            System.out.printf("Sample %d: Winning probability = %.4f%n", i + 1, prob);
        }
    }
}
