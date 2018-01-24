package de.doubleslash.ml.techdays;

import java.util.Arrays;
import java.util.List;

import org.apache.spark.ml.clustering.KMeans;
import org.apache.spark.ml.clustering.KMeansModel;
import org.apache.spark.ml.feature.Normalizer;
import org.apache.spark.ml.linalg.DenseVector;
import org.apache.spark.ml.linalg.Vector;
import org.apache.spark.ml.linalg.VectorUDT;
import org.apache.spark.ml.linalg.Vectors;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.Metadata;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;

public class ClusteringExample {
	private static final double SQM_MUC = 73;
	private static final double PRICE_MUC = 1000;

	private static final double SQM_FN = 69;
	private static final double PRICE_FN = 795;
	private static Normalizer normalizer;
	private static SparkSession spark;

	public static void main(String args[]) {
		initSparkSession();

		// Read data from disk
		Dataset<Row> trainingData = spark.read().format("libsvm").load("data/clustering.txt");

		// Normalize trainingData
		KMeansModel normalizedTrainingModel = getNormalizedKMeansModel(trainingData);

		showClusterCenters(normalizedTrainingModel);

		predictData(normalizedTrainingModel, SQM_MUC, PRICE_MUC);
		predictData(normalizedTrainingModel, SQM_FN, PRICE_FN);
	}

	private static void initSparkSession() {
		spark = SparkSession.builder().appName("Java Spark SQL basic example").config("spark.master", "local")
				.getOrCreate();
		spark.sparkContext().setLogLevel("ERROR");
	}

	private static KMeansModel getNormalizedKMeansModel(Dataset<Row> dataset) {
		// Initialize Normalizer
		normalizer = new Normalizer().setInputCol("features").setOutputCol("normFeatures").setP(1.0);

		// Normalize training data
		Dataset<Row> normalizedDataset = normalizer.transform(dataset);
		normalizedDataset.show(false);
		KMeans kmeans = new KMeans().setK(2).setMaxIter(50).setFeaturesCol("normFeatures");

		return kmeans.fit(normalizedDataset);
	}

	private static DenseVector getTestDataAsNormalizedVector(double sqm, double price) {
		Vector testData = Vectors.dense(sqm, price);
		List<Row> dataList = Arrays.asList(RowFactory.create(0, testData));
		StructType schema = new StructType(
				new StructField[] { new StructField("id", DataTypes.IntegerType, false, Metadata.empty()),
						new StructField("features", new VectorUDT(), false, Metadata.empty()) });
		Dataset<Row> testDataFrame = spark.createDataFrame(dataList, schema);

		Dataset<Row> normalizedTestDataset = normalizer.transform(testDataFrame);

		Row featuresRow = normalizedTestDataset.select("normFeatures").first();
		DenseVector normalizedVector = (DenseVector) featuresRow.get(0);
		return normalizedVector;
	}

	private static void showClusterCenters(KMeansModel normalizedTrainingModel) {
		Vector[] centers = normalizedTrainingModel.clusterCenters();
		System.out.println("Cluster Centers: ");
		int clusterNr = 0;
		for (Vector center : centers) {
			System.out.println("Cluster " + clusterNr++ + ":" + center);
		}
	}

	private static void predictData(KMeansModel normalizedTrainingModel, double sqm, double price) {
		DenseVector normalizedVector = getTestDataAsNormalizedVector(sqm, price);
		System.out.println("Prediction for " + sqm + "m² and " + price + "€: "
				+ normalizedTrainingModel.predict(normalizedVector));
	}
}
