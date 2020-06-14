package com.demoyer.neuralnetwork.tut.controller;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.conf.BackpropType;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

import java.io.File;

//created using baeldung tutorial with minor adjustments, used to learn how various algorithmic tweaks affect training the model
@RestController
public class NeuralNetworkController {

    private static final int CLASSES = 2;
    private static final int FEATURES = 5;
    private static final int totalEntriesInCSV = 2507;

    @GetMapping("/performCalculation")
    public void performCalculation(@RequestParam("fileName") String fileName) {


        DataSet dataSet = new DataSet();
        try {
            RecordReader recordReader = new CSVRecordReader(1, ',');
            recordReader.initialize(new FileSplit(new File("C://CSVOutput/" + fileName + ".csv")));

            DataSetIterator iterator = new RecordReaderDataSetIterator(recordReader, totalEntriesInCSV, 5, CLASSES);
            dataSet = iterator.next();
        } catch (Exception e) {
            e.printStackTrace();
        }

        DataNormalization normalizer = new NormalizerStandardize();
        normalizer.fit(dataSet);
        normalizer.transform(dataSet);

        SplitTestAndTrain testAndTrain = dataSet.splitTestAndTrain(0.65);
        DataSet trainingData = testAndTrain.getTrain();
        DataSet testData = testAndTrain.getTest();

        MultiLayerConfiguration configuration = new NeuralNetConfiguration.Builder()
                .iterations(1000)
                .activation(Activation.SIGMOID)
                .weightInit(WeightInit.XAVIER)
                .regularization(true)
                .learningRate(0.45).l2(0.0001)
                .list()
                .layer(0, new DenseLayer.Builder().nIn(FEATURES).nOut(3)
                        .build())
                .layer(1, new DenseLayer.Builder().nIn(3).nOut(3)
                        .build())
                .layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.XENT)
                        .activation(Activation.SOFTMAX)
                        .nIn(3).nOut(CLASSES).build())
                .backpropType(BackpropType.Standard).pretrain(false)
                .build();

        MultiLayerNetwork model = new MultiLayerNetwork(configuration);
        model.init();
        model.fit(trainingData);

        INDArray output = model.output(testData.getFeatures());

        Evaluation eval = new Evaluation(CLASSES);
        eval.eval(testData.getLabels(), output);
        System.out.println(eval.stats());
    }
}
