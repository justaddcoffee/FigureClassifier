package org.deeplearning4j.examples.convolution;

import org.apache.commons.io.FilenameUtils;
import org.datavec.api.io.filters.BalancedPathFilter;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.datavec.image.transform.FlipImageTransform;
import org.datavec.image.transform.ImageTransform;
import org.datavec.image.transform.PipelineImageTransform;
import org.datavec.image.transform.WarpImageTransform;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.nd4j.evaluation.classification.Evaluation;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.primitives.Pair;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

import static java.lang.Math.toIntExact;

/**
 * This software classifies figures taken from a primary literature paper
 * into either "picture" (e.g. microscopy image of a worm, immunofluorescence image, etc)
 * or a "non-picture" (e.g. a graph, a barchart, a flowchart)
 *
 * The canonical input for this software is images output from the analyze command of
 * figtools:
 * https://github.com/INCATools/figtools
 *
 * <p>
 * Cribbed from the AnimalsClassification.java example code:
 * CHALLENGE: Current setup gets low score results. Can you improve the scores? Some approaches:
 * - Add additional images to the dataset
 * - Apply more transforms to dataset
 * - Increase epochs
 * - Try different model configurations
 * - Tune by adjusting learning rate, updaters, activation & loss functions, regularization, ...
 */

public class FigureClassification {

    private void run(String[] args) throws Exception {

        final Logger log = LoggerFactory.getLogger(FigureClassification.class);
        int height = 100;
        int width = 100;
        int channels = 3;
        int batchSize = 20;

        long seed = 42;
        Random rng = new Random(seed);
        int epochs = 50;
        double splitTrainTest = 0.8;
        int maxPathsPerLabel = 18;

        String modelType = "AlexNet"; // LeNet, AlexNet or Custom but you need to fill it out
        int numLabels;

        log.info("Load data....");

         /*
         * Data Setup -> organize and limit data file paths:
         *  - trainingImages = path to image files
         *  - fileSplit = define basic dataset split with limits on format
         *  - pathFilter = define additional file load filter to limit size and balance batch content
         */

        ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();
        // TODO: get directory for imagesToBeClassified and trainingImages from command line args
        File trainingImages = new File(System.getProperty("user.dir"), "src/main/resources/training_data/");
        File imagesToBeClassified = new File(System.getProperty("user.dir"),
                "src/main/resources/images_to_classify/");

        if (!trainingImages.exists()) {
            throw new RuntimeException("path to training data $" + trainingImages.getAbsolutePath() + "does not exist");
        }
        if (!imagesToBeClassified.exists()) {
            throw new RuntimeException("path to images to be classified $" + imagesToBeClassified.getAbsolutePath() +
                    "does not exist");
        }

        FileSplit fileSplit = new FileSplit(trainingImages, NativeImageLoader.ALLOWED_FORMATS, rng);
        int numExamples = toIntExact(fileSplit.length());

        //This only works if your root is clean: only label subdirs.
        numLabels = fileSplit.getRootDir().listFiles(File::isDirectory).length;

        if (numLabels != 2) {
            System.out.println("training image dir:\n`" + trainingImages.getAbsolutePath() +
            "\nshould have exactly two subdirectories with picture/ and nonpicture/ data" +
                    "\n(found " + numLabels + " directories)");
            System.exit(-1);
        }

        assert numLabels==2 : "Expected 2 directories in trainingImages (hopefully pictures/ and nonpictures/)";
        BalancedPathFilter pathFilter = new BalancedPathFilter(rng, labelMaker, numExamples, numLabels, maxPathsPerLabel);

        /*
         * Data Setup -> train test split
         *  - inputSplit = define train and test split
        */
        InputSplit[] inputSplit = fileSplit.sample(pathFilter, splitTrainTest, 1 - splitTrainTest);
        InputSplit trainData = inputSplit[0];
        InputSplit testData = inputSplit[1];

        /*
         * Data Setup -> transformation
         *  - Transform = how to tranform images and generate large dataset to train on
        */
        ImageTransform flipTransform1 = new FlipImageTransform(rng);
        ImageTransform flipTransform2 = new FlipImageTransform(new Random(123));
        ImageTransform warpTransform = new WarpImageTransform(rng, 42);
        boolean shuffle = false;
        List<Pair<ImageTransform, Double>> pipeline = Arrays.asList(new Pair<>(flipTransform1, 0.9),
                new Pair<>(flipTransform2, 0.8),
                new Pair<>(warpTransform, 0.5));

        ImageTransform transform = new PipelineImageTransform(pipeline, shuffle);
        /*
         * Data Setup -> normalization
         *  - how to normalize images and generate large dataset to train on
        */
        DataNormalization scaler = new ImagePreProcessingScaler(0, 1);

        log.info("Building model....");

        // Uncomment below to try AlexNet. Note change height and width to at least 100
        // MultiLayerNetwork network = new AlexNet(height, width, channels, numLabels, seed, iterations).init();

        MultiLayerNetwork network = ModelGenerator.getModelForType(modelType, numLabels);
        network.init();
        // network.setListeners(new ScoreIterationListener(listenerFreq));
        UIServer uiServer = UIServer.getInstance();
        StatsStorage statsStorage = new InMemoryStatsStorage();
        uiServer.attach(statsStorage);
        network.setListeners(new StatsListener(statsStorage), new ScoreIterationListener(1));
        /*
         * Data Setup -> define how to load data into net:
         *  - recordReader = the reader that loads and converts image data pass in inputSplit to initialize
         *  - dataIter = a generator that only loads one batch at a time into memory to save memory
         *  - trainIter = uses MultipleEpochsIterator to ensure model runs through the data for all epochs
        */
        ImageRecordReader recordReader = new ImageRecordReader(height, width, channels, labelMaker);
        DataSetIterator dataIter;


        log.info("Training model....");
        // Train without transformations
        recordReader.initialize(trainData, null);
        dataIter = new RecordReaderDataSetIterator(recordReader, batchSize, 1, numLabels);
        scaler.fit(dataIter);
        dataIter.setPreProcessor(scaler);
        network.fit(dataIter, epochs);

        // Train with transformations
        recordReader.initialize(trainData, transform);
        dataIter = new RecordReaderDataSetIterator(recordReader, batchSize, 1, numLabels);
        scaler.fit(dataIter);
        dataIter.setPreProcessor(scaler);
        network.fit(dataIter, epochs);

        log.info("Evaluating model....");
        recordReader.initialize(testData);
        dataIter = new RecordReaderDataSetIterator(recordReader, batchSize, 1, numLabels);
        scaler.fit(dataIter);
        dataIter.setPreProcessor(scaler);
        Evaluation eval = network.evaluate(dataIter);
        log.info(eval.stats(true));

        // classify things in imagesToBeClassified
        ImageRecordReader recReader = new ImageRecordReader(height, width, channels);
        recReader.initialize(new FileSplit(new File(imagesToBeClassified.getAbsolutePath())));
        DataSetIterator iter = new RecordReaderDataSetIterator(recReader, 1);

        while (iter.hasNext()) {
            DataSet testDataSet = iter.next();
            List<String> allClassLabels = recordReader.getLabels();
            int[] predictedClasses = network.predict(testDataSet.getFeatures());
            String modelPrediction = allClassLabels.get(predictedClasses[0]);
            System.out.print(
                    "model classifies this file\n:" +
                    recReader.getCurrentFile().toString() +
                    "\nas: " + modelPrediction + "\n\n");
        }

        log.info("Save model....");
        String basePath = FilenameUtils.concat(System.getProperty("user.dir"), "src/main/resources/");
        ModelSerializer.writeModel(network, basePath + "model.bin", true);

        log.info("****************finished********************");
    }

    public static void main(String[] args) throws Exception {
        new FigureClassification().run(args);
    }

}
