package learn;

import de.bwaldvogel.liblinear.*;
import statistical.ScoreDict;
import utilities.FileIO;
import utilities.Logger;

import java.io.*;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashSet;
import java.util.List;

/*
 *
 */
public class LiblinearSVM
{
    private Parameter _params;
    private SolverType _solverType;
    private Model _model;

    public LiblinearSVM(String solverTypeName, double cost, double epsilon)
    {
        _solverType = SolverType.valueOf(solverTypeName);
        _params = new Parameter(_solverType, cost, epsilon);
    }

    public LiblinearSVM(String filename)
    {
        Logger.log("Loading _model from file");
        try {
            _model = Model.load(new File(filename));
        } catch(IOException ioEx) {
            Logger.log(ioEx);
        }
    }

    public boolean getIsProbabilityModel()
    {return _model.isProbabilityModel();}

    public void train(String filename)
    {
        train(filename, new ArrayList<>(), new HashSet<>());
    }

    public void train_excludeIndices(String filename, Collection<Integer> ignoredIndices)
    {
        train(filename, ignoredIndices, new HashSet<>());
    }

    @Deprecated
    public void train_excludeIndices(Collection<FeatureVector> featureVectors, Collection<Integer> ignoredIndices)
    {
        _model = null;
        Problem problem = new Problem();
        int numVectors = featureVectors.size();
        Feature[][] x;
        double[] y;
        int maxFeatIdx = 0;

        //read the file into feature vectors
        Logger.log("Reading feature vectors into liblinear structures");
        problem.l = numVectors;
        x = new Feature[numVectors][];
        y = new double[numVectors];
        int idx = 0;

        for(FeatureVector fv : featureVectors){
            y[idx] = fv.label;
            List<Integer> featureIndices =
                    fv.getFeatureIndices();
            List<Feature> featureList = new ArrayList<>();
            int j=0;
            for(int fIdx : featureIndices){
                if(!ignoredIndices.contains(fIdx)){
                    featureList.add(new FeatureNode(fIdx, fv.getFeatureValue(fIdx)));
                    if(fIdx > maxFeatIdx)
                        maxFeatIdx = fIdx;
                    j++;
                }
            }
            Feature[] fArr = new Feature[featureList.size()];
            x[idx] = featureList.toArray(fArr);
            idx++;
        }

        //set the problem vars
        problem.n = maxFeatIdx;
        problem.x = x;
        problem.y = y;

        //Finally, train_excludeIndices the _model
        Logger.log("Training model");
        _model = Linear.train(problem, _params);
    }

    public void train_excludeIDs(String filename, Collection<String> ignoredIDs)
    {
        train(filename, new ArrayList<>(), ignoredIDs);
    }

    private void train(String filename, Collection<Integer> ignoredIndices,
                       Collection<String> ignoredIDs)
    {
        _model = null;
        Problem problem = new Problem();
        int numVectors = 0;
        int maxFeatIdx = 0;
        try{
            Logger.log("Getting num vectors from file");
            BufferedReader br = new BufferedReader(
                    new InputStreamReader(new FileInputStream(filename)));
            while (br.readLine() != null) numVectors++;
            br.close();
            br = new BufferedReader(
                    new InputStreamReader(new FileInputStream(filename)));

            //read the file into feature vectors
            Logger.log("Reading feature file [" + filename +
                    "] into liblinear structures");

            List<Feature[]> xList = new ArrayList<>();
            List<Double> yList = new ArrayList<>();
            String nextLine = br.readLine();
            int idx = 0;
            while(nextLine != null){
                Logger.logStatus("%.2f%% complete", 100.0 * idx / numVectors);
                FeatureVector fv = FeatureVector.parseFeatureVector(nextLine);
                String ID = fv.comments;
                if(!ignoredIDs.contains(ID)){
                    yList.add(fv.label);
                    List<Integer> featureIndices =
                            fv.getFeatureIndices();
                    List<Feature> featureList = new ArrayList<>();
                    int j=0;
                    for(int fIdx : featureIndices){
                        if(!ignoredIndices.contains(fIdx)){
                            featureList.add(new FeatureNode(fIdx, fv.getFeatureValue(fIdx)));
                            if(fIdx > maxFeatIdx)
                                maxFeatIdx = fIdx;
                            j++;
                        }
                    }
                    Feature[] fArr = new Feature[featureList.size()];
                    xList.add(featureList.toArray(fArr));

                }
                idx++;
                nextLine = br.readLine();
            }
            br.close();

            //cast x as an array
            Feature[][] x = new Feature[xList.size()][];
            x = xList.toArray(x);

            //down convert to primitives
            double[] y = new double[yList.size()];
            for(int i=0; i<yList.size(); i++)
                y[i] = yList.get(i);

            //set the problem vars
            problem.l = yList.size();
            problem.n = maxFeatIdx;
            problem.x = x;
            problem.y = y;
        } catch(IOException ioEx){
            Logger.log(ioEx);
        }

        //Finally, train_excludeIndices the _model
        Logger.log("Training model");
        _model = Linear.train(problem, _params);
    }

    public void saveModel(String filename)
    {
        File f = new File(filename);
        try{
            _model.save(f);
        } catch(IOException ioEx) {
            Logger.log(ioEx);
        }
    }

    public void saveScores(String featureFile)
    {
        Logger.log("Calculating scores");
        BinaryClassifierScoreDict scoreDict =
                new BinaryClassifierScoreDict();
        try{
            BufferedReader br = new BufferedReader(
                    new InputStreamReader(new FileInputStream(featureFile)));
            String nextLine = br.readLine();
            while(nextLine != null){
                FeatureVector fv = FeatureVector.parseFeatureVector(nextLine);
                double y = fv.label;
                double[] probabilities = new double[2];
                double yhat = predict(fv, probabilities);
                scoreDict.addScore(fv.comments,
                        BinaryClassifierScoreDict.getNormScore((int)yhat, probabilities));
                nextLine = br.readLine();
            }
            br.close();
        } catch(IOException ioEx) {
            Logger.log(ioEx);
        }

        Logger.log("Writing scores to [" +featureFile.replace(".feats", ".scores") + "]");
        FileIO.writeFile(scoreDict, featureFile.replace(".feats", ""), "scores", false);
        Logger.log("done");
    }

    public double predict(FeatureVector featVector)
    {
        Feature[] instance = new Feature[featVector.getFeatureIndices().size()];
        int i = 0;
        for(int idx : featVector.getFeatureIndices()) {
            instance[i] = new FeatureNode(idx, featVector.getFeatureValue(idx));
            i++;
        }
        return Linear.predict(_model, instance);
    }

    public double predict(FeatureVector featVector, double[] probabilities)
    {
        Feature[] instance = new Feature[featVector.getFeatureIndices().size()];
        int i = 0;
        for(int idx : featVector.getFeatureIndices()) {
            instance[i] = new FeatureNode(idx, featVector.getFeatureValue(idx));
            i++;
        }
        return Linear.predictProbability(_model, instance, probabilities);
    }

    public void evaluate(String filename)
    {
        evaluate_excludeIDs(filename, new HashSet<>());
    }

    public void evaluate_excludeIDs(String filename, Collection<String> ignoredIDs)
    {
        int numVectors = 0;
        Logger.log("Getting num vectors from file");
        try {
            BufferedReader br = new BufferedReader(
                    new InputStreamReader(new FileInputStream(filename)));
            while (br.readLine() != null) numVectors++;
            br.close();
        } catch (Exception ex){
            Logger.log(ex);
        }

        ScoreDict<Integer> scoreDict = new ScoreDict<>();
        try{
            BufferedReader br = new BufferedReader(
                    new InputStreamReader(new FileInputStream(filename)));
            String nextLine = br.readLine();
            int idx = 0;
            while(nextLine != null){
                Logger.logStatus("%.2f%% complete", 100.0 * idx / numVectors);
                FeatureVector fv = FeatureVector.parseFeatureVector(nextLine);
                if(!ignoredIDs.contains(fv.comments)){
                    double y = fv.label;
                    double yhat = predict(fv);
                    scoreDict.increment((int)y, (int)yhat);
                }
                nextLine = br.readLine();
                idx++;
            }
            br.close();
        }catch(IOException ioEx){
            Logger.log(ioEx);
        }

        System.out.println("Pos: " + scoreDict.getScore(1).toScoreString());
        System.out.println("Neg: " + scoreDict.getScore(0).toScoreString());
        System.out.println("Acc: " + scoreDict.getAccuracy() + " (of "+ scoreDict.getTotalGold() + ")");
    }
}
