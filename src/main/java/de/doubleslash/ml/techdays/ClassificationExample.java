package de.doubleslash.ml.techdays;

import org.apache.spark.ml.classification.LinearSVC;
import org.apache.spark.ml.classification.LinearSVCModel;
import org.apache.spark.ml.linalg.Vector;
import org.apache.spark.ml.linalg.Vectors;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

public class ClassificationExample {
	public static void main(String[] args) {
		SparkSession spark = SparkSession.builder().appName("JavaLinearSVCExample").config("spark.master", "local")
				.getOrCreate();
		spark.sparkContext().setLogLevel("ERROR");

		// Load training data
		Dataset<Row> training = spark.read().format("libsvm").load("data/mietkosten_fn_muc.txt");

		LinearSVC lsvc = new LinearSVC().setMaxIter(10).setRegParam(0.1);

		// Fit the model
		LinearSVCModel lsvcModel = lsvc.fit(training);

		// Print the coefficients and intercept for LinearSVC
		System.out.println("Coefficients: " + lsvcModel.coefficients() + " Intercept: " + lsvcModel.intercept());

		Vector testData = Vectors.dense(2100, 102);
		System.out.println("Prediction: " + lsvcModel.predict(testData));

		spark.stop();
	}
}
