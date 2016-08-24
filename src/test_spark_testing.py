import os
import sys
import unittest

try:
    # Append PySpark to PYTHONPATH / Spark 2.0.0
    sys.path.append(os.path.join(os.environ['SPARK_HOME'], "python"))
    sys.path.append(os.path.join(os.environ['SPARK_HOME'], "python", "lib", "py4j-0.10.1-src.zip"))
except KeyError as e:
    print("SPARK_HOME is not set", e)
    sys.exit(1)

try:
    # Import PySpark modules here
    from pyspark import SparkConf
    from pyspark.context import SparkContext
    from pyspark.sql import SparkSession
except ImportError as e:
    print("Can not import Spark modules", e)
    sys.exit(1)

import spark_testing

class SparkTest(unittest.TestCase):

    def setUp(self):
        """Create a single node Spark application."""
        conf = SparkConf()
        conf.set("spark.executor.memory", "1g")
        conf.set("spark.cores.max", "1")
        conf.set("spark.app.name", "nosetest")
        SparkSession._instantiatedContext = None
        self.spark = SparkSession.builder.config(conf=conf).getOrCreate()
        self.sc = self.spark.sparkContext
        self.mock_df = self.mock_data()


    def mock_data(self):
        mock_data_rdd = self.sc.parallelize([('Alabama', 'Dat', 'Democrat', 80, 0.8),
                                             ('Alabama', 'Batman', 'Democrat', 20, 0.2),
                                             ('California', 'Dat', 'Democrat', 70, 0.7),
                                             ('California', 'Batman', 'Democrat', 30, 0.3),
                                             ('Florida', 'Dat', 'Democrat', 80, 0.8),
                                             ('Florida', 'Batman', 'Democrat', 20, 0.2),
                                             ('Alabama', 'Donald', 'Republican', 80, 0.8),
                                             ('Alabama', 'Superman', 'Republican', 20, 0.2),
                                             ('California', 'Donald', 'Republican', 70, 0.7),
                                             ('California', 'Superman', 'Republican', 30, 0.3),
                                             ('Florida', 'Donald', 'Republican', 80, 0.8),
                                             ('Florida', 'Superman', 'Republican', 20, 0.2)
                                            ])
        schema = ['state', 'candidate', 'party', 'votes', 'fraction_votes' ]
        mock_data_df = self.spark.createDataFrame(mock_data_rdd, schema)
        return mock_data_df

    def tearDown(self):
        """Stop the SparkContext."""
        self.spark.stop()
        self.sc.stop()

    def test_count(self):
        """Check if data has 12 rows."""
        self.assertEqual(self.mock_df.count(), 12)

    def test_groupby(self):
        """Check groupby."""
        grouped_df = spark_testing.grouped_data(self.mock_df)
        self.assertEqual(grouped_df.count(), 3)

    def test_model_acc(self):
        """Check the accuracy_score of the model."""
        grouped_df = spark_testing.grouped_data(self.mock_df)
        output = spark_testing.model_output(grouped_df)
        state, acc = output.take(1)[0]
        self.assertEqual(acc, 1.0)
