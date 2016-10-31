package learn;

import statistical.ScoreDict;
import utilities.DoubleDict;
import utilities.FileIO;
import utilities.Logger;
import utilities.StringUtil;
import weka.classifiers.functions.Logistic;
import weka.classifiers.meta.MultiClassClassifierUpdateable;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SelectedTag;
import weka.core.converters.ArffLoader;

import java.io.*;
import java.util.*;

public class WekaMulticlass
{
    private static final int _PAIRWISE_IDX_LIMIT = 1050823; //actual max token idx: 1050822
    private static final int _PAIRWISE_TRAIN_EXAMPLES = (int)(1.5*10e6);
    private static final int _PAIRWISE_DEV_EXAMPLES = (int)(1.8*10e4);
    private static final String[] _PAIRWISE_LABELS = {"N", "C", "S"};
    private static final double[] _PAIRWISE_LABEL_DISTRO = {0.8, 0.15, 0.05};

    private MultiClassClassifierUpdateable _mcc;

    /**Constructor for multiclass classifier to be trained
     */
    public WekaMulticlass()
    {
        _mcc = new MultiClassClassifierUpdateable();
        _mcc.setMethod(new SelectedTag(MultiClassClassifierUpdateable.METHOD_1_AGAINST_1,
                MultiClassClassifierUpdateable.TAGS_METHOD));
    }

    /**Constructor that loads multiclass classifier from model file
     *
     * @param modelFile - the model file to load the classifier from
     */
    public WekaMulticlass(String modelFile)
    {
        try{
            _mcc = (MultiClassClassifierUpdateable)(new ObjectInputStream(new FileInputStream(modelFile)).readObject());
        } catch(Exception ex){
            Logger.log("Couldn't load model from " + modelFile);
            Logger.log(ex);
        }
    }

    public void _train(String featureFile, String modelFile)
    {
        try{
            Logger.tic("Loading dataset");
            ArffLoader loader = new ArffLoader();
            loader.setFile(new File(featureFile));
            Instances dataset = loader.getDataSet();
            Instances structure = loader.getStructure();
            dataset.setClassIndex(dataset.numAttributes()-1);
            structure.setClassIndex(structure.numAttributes()-1);
            Logger.toc();


            Logger.tic("Training");
            Logistic _log = new Logistic();
            _log.buildClassifier(dataset);
            Logger.toc();

            Logger.log("Saving the model");
            ObjectOutputStream oos = new ObjectOutputStream(
                    new FileOutputStream(modelFile));
            oos.writeObject(_mcc);
            oos.flush();
            oos.close();
        } catch(Exception ex){
            Logger.log("Failed to train and save model with " + featureFile);
            Logger.log(ex);
        }
    }

    public void _eval(String featureFile, String modelFile)
    {
        Logistic _log = new Logistic();
        try{
            _log = (Logistic)(new ObjectInputStream(
                    new FileInputStream(modelFile)).readObject());
        } catch(Exception ex){
            Logger.log("Couldn't load model from " + modelFile);
            Logger.log(ex);
        }

        try {
            Logger.tic("Loading dataset");
            ArffLoader loader = new ArffLoader();
            loader.setFile(new File(featureFile));
            Instances dataset = loader.getDataSet();
            dataset.setClassIndex(dataset.numAttributes()-1);
            Logger.toc();

            Logger.log("Evaluating");
            ScoreDict<String> scoreDict = new ScoreDict<>();
            for(int i=0; i<dataset.numInstances(); i++){
                Instance inst = dataset.get(i);
                double[] dist = _log.distributionForInstance(inst);
                double max = -1;
                String predLabel = null;
                for(int j=0; j<dist.length; j++){
                    if(dist[j] > max){
                        max = dist[j];
                        predLabel = _PAIRWISE_LABELS[j];
                    }
                }
                String goldLabel = _PAIRWISE_LABELS[(int)inst.classValue()];
                scoreDict.increment(goldLabel, predLabel);
                Logger.logStatus("Seen %d instances (%.2f%%)", i,
                        100.0 * (double)i / dataset.numInstances());
            }

            System.out.printf("Acc: %.2f%%\n", scoreDict.getAccuracy());
            for(String label : scoreDict.keySet())
                System.out.println(label + " & " + scoreDict.getScore(label).toLatexString());
            System.out.println("True distro");
            for(String label : scoreDict.keySet())
                System.out.printf("%s: %d (%.2f%%)\n", label, scoreDict.getGoldCount(label),
                        100.0 * scoreDict.getGoldCount(label) / scoreDict.getTotalGold());
        } catch(Exception ex){
            Logger.log(ex);
            ex.printStackTrace();
        }
    }

    /**Trains the internal multiclass classifier with the ARFF file
     * specified by filename; saves the model in modelFile
     *
     * @param featureFile - Feature file in ARFF format
     * @param modelFile   - File to write the model in
     */
    public void train(String featureFile, String modelFile,
                      int batchSize, int numEpochs, boolean balance)
    {
        try{
            Logger.tic("Loading dataset");
            ArffLoader loader = new ArffLoader();
            loader.setFile(new File(featureFile));
            Instances dataset = loader.getDataSet();
            Instances structure = loader.getStructure();
            dataset.setClassIndex(dataset.numAttributes()-1);
            structure.setClassIndex(structure.numAttributes()-1);
            Logger.toc();

            Logger.tic("Reshaping dataset");
            Map<String, LinkedList<Instance>> instanceDict = new HashMap<>();
            for(String label : _PAIRWISE_LABELS)
                instanceDict.put(label, new LinkedList<>());
            for(int i=0; i<dataset.numInstances(); i++){
                Instance inst = dataset.get(i);
                String label = _PAIRWISE_LABELS[(int)inst.classValue()];
                instanceDict.get(label).add(inst);
            }
            Logger.toc();

            Logger.log("Setting up classifier");
            _mcc = new MultiClassClassifierUpdateable();
            _mcc.setUsePairwiseCoupling(true);
            _mcc.buildClassifier(structure);

            Logger.log("Training");
            for(int i=0; i<numEpochs; i++){
                Logger.logStatus("Epoch: %d", i);
                LinkedList<Instance> batch = new LinkedList<>();
                if(balance){
                    for(int j=0; j < _PAIRWISE_LABELS.length; j++){
                        String label = _PAIRWISE_LABELS[j];
                        int batchSize_bal = (int)(_PAIRWISE_LABEL_DISTRO[j] * batchSize);
                        Collections.shuffle(instanceDict.get(label));
                        batch.addAll(instanceDict.get(label).subList(0, batchSize_bal));
                    }
                } else {
                    for(String label : instanceDict.keySet())
                        batch.addAll(instanceDict.get(label));
                    Collections.shuffle(batch);
                    batch = new LinkedList<>(batch.subList(0, batchSize));
                }
                Collections.shuffle(batch);

                for(Instance inst : batch)
                    _mcc.updateClassifier(inst);

                if((i+1) % 1000 == 0){
                    Logger.log("Saving the model at %d epochs", i);
                    ObjectOutputStream oos = new ObjectOutputStream(
                            new FileOutputStream(modelFile));
                    oos.writeObject(_mcc);
                    oos.flush();
                    oos.close();
                }
            }
            Logger.log("Saving the final model");
            ObjectOutputStream oos = new ObjectOutputStream(
                    new FileOutputStream(modelFile));
            oos.writeObject(_mcc);
            oos.flush();
            oos.close();


/*
            System.out.println("tic"); tic = System.currentTimeMillis();
            Instances dataset = loader.getDataSet();
            System.out.printf("toc: %.3f\n", (System.currentTimeMillis() - tic) / 1000.0);
            System.out.println("tic"); tic = System.currentTimeMillis();
             RemoveType idFilter = new RemoveType();
             idFilter.setInputFormat(loader.getStructure());
             dataset = Filter.useFilter(dataset, idFilter);
             System.out.printf("toc: %.3f\n", (System.currentTimeMillis() - tic) / 1000.0);

            dataset.setClassIndex(dataset.numAttributes()-1);
*/

            /*
            Instances structure = loader.getStructure();
            structure.setClassIndex(structure.numAttributes() - 1);*/
/*
            Logger.log("Setting up classifier");
            _mcc = new MultiClassClassifierUpdateable();

            Logger.log("Training");
            System.out.println("tic"); tic = System.currentTimeMillis();
            _mcc.buildClassifier(dataset);
            System.out.printf("toc: %.3f\n", (System.currentTimeMillis() - tic) / 1000.0);
*/
            /*
            Logger.log("Training");
            //_mcc.setUsePairwiseCoupling(true);
            _mcc.setDebug(true);
            _mcc.buildClassifier(dataset);
            */

            /*
            Logger.log("Training");
            Instance current = loader.getNextInstance(structure);
            int idx = 0;
            while(current != null){
                current.setDataset(structure);
                System.out.println("tic (update)"); tic = System.currentTimeMillis();
                _mcc.updateClassifier(current);
                System.out.printf("toc: %.3f\n", (System.currentTimeMillis() - tic) / 1000.0);


                System.out.println("tic (update)"); tic = System.currentTimeMillis();
                current = loader.getNextInstance(structure);
                System.out.printf("toc: %.3f\n", (System.currentTimeMillis() - tic) / 1000.0);
                idx++;
            }
            */
            /*
            while((current = loader.getNextInstance(structure)) != null){
                current.setDataset(structure);
                _mcc.updateClassifier(current);
                idx++;
                Logger.logStatus("%.2f%% complete", 100.0 * (double)idx / _PAIRWISE_TRAIN_EXAMPLES);
            }*/


            //Instances structure = loader.getStructure();
            //Remove idFilter = new Remove();
            //idFilter.setAttributeIndicesArray(new int[]{_PAIRWISE_IDX_LIMIT-1});
            //structure.setClassIndex(structure.numAttributes() - 1);
        } catch(Exception ex){
            Logger.log("Failed to train and save model with " + featureFile);
            Logger.log(ex);
        }
    }

    /**Evaluates this classifier using the ARFF evaluate
     * file specified by featureFile
     *
     * @param featureFile - Feature file to evaluate
     */
    public void evaluate(String featureFile)
    {
        try {
            Logger.tic("Loading dataset");
            ArffLoader loader = new ArffLoader();
            loader.setFile(new File(featureFile));
            Instances dataset = loader.getDataSet();
            dataset.setClassIndex(dataset.numAttributes()-1);
            Logger.toc();


            Logger.log("Evaluating");
            ScoreDict<String> scoreDict = new ScoreDict<>();
            for(int i=0; i<dataset.numInstances(); i++){
                Instance inst = dataset.get(i);
                double[] dist = _mcc.distributionForInstance(inst);
                double max = -1;
                String predLabel = null;
                for(int j=0; j<dist.length; j++){
                    if(dist[j] > max){
                        max = dist[j];
                        predLabel = _PAIRWISE_LABELS[j];
                    }
                }
                String goldLabel = _PAIRWISE_LABELS[(int)inst.classValue()];
                scoreDict.increment(goldLabel, predLabel);
                Logger.logStatus("Seen %d instances (%.2f%%)", i,
                        100.0 * (double)i / dataset.numInstances());
            }

            System.out.printf("Acc: %.2f%%\n", scoreDict.getAccuracy());
            for(String label : scoreDict.keySet())
                System.out.println(label + " & " + scoreDict.getScore(label).toLatexString());
            System.out.println("True distro");
            for(String label : scoreDict.keySet())
                System.out.printf("%s: %d (%.2f%%)\n", label, scoreDict.getGoldCount(label),
                        100.0 * scoreDict.getGoldCount(label) / scoreDict.getTotalGold());
        } catch(Exception ex){
            Logger.log(ex);
            ex.printStackTrace();
        }
    }

    /**Convert a sparse .feats file to a sparse .arff file
     *
     * @param featsFile - File to convert
     */
    public static void exportToArff(String featsFile)
    {
        Logger.log("Setting out arff headers");
        List<String> outList = new ArrayList<>();
        outList.add("@relation Flickr30kEntities_v2");
        outList.add("");
        for(int i=1; i<_PAIRWISE_IDX_LIMIT; i++)
            outList.add("@attribute f_" + i + " numeric");
        outList.add("@attribute ID string");
        outList.add("@attribute Class {N, C, S}");
        outList.add("");
        outList.add("@data");

        Logger.log("Reading feature vectors from " + featsFile);
        Set<FeatureVector> fvSet = new HashSet<>();
        for (String line : FileIO.readFile_lineList(featsFile))
            fvSet.add(FeatureVector.parseFeatureVector(line));
        DoubleDict<Integer> labelDict = new DoubleDict<>();
        for (FeatureVector fv : fvSet)
            labelDict.increment((int) fv.label);
        List<String> labelDistro = new ArrayList<>();
        for (Integer label : labelDict.keySet()){
            labelDistro.add(String.format("%d: %.2f%%", label,
                    100.0 * labelDict.get(label) / labelDict.getSum()));
        }
        Logger.log("Label distro: " + StringUtil.listToString(labelDistro, "; "));

        Logger.log("Converting feature vectors to arff format");
        for(FeatureVector fv : fvSet){
            StringBuilder sb = new StringBuilder();
            sb.append("{");
            for(int idx : fv.getFeatureIndices()){
                sb.append(idx-1);   //for arf, indices start at 0
                sb.append(" ");
                sb.append(fv.getFeatureValue(idx));
                sb.append(", ");
            }
            sb.append(_PAIRWISE_IDX_LIMIT-1);
            sb.append(" \'");
            sb.append(fv.comments);
            sb.append("\', ");
            sb.append(_PAIRWISE_IDX_LIMIT);
            sb.append(" ");
            sb.append(_PAIRWISE_LABELS[(int)fv.label]);
            sb.append("}");
            outList.add(sb.toString());
        }
        FileIO.writeFile(outList, featsFile.replace(".feats", "_with_IDs"), "arff", false);
    }
}
