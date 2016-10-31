package learn;

import out.HtmlScribe;
import statistical.BCubed;
import statistical.Blanc;
import statistical.Score;
import structures.AttrStruct;
import structures.Chain;
import structures.Document;
import structures.Mention;
import utilities.FileIO;
import utilities.Logger;

import java.util.*;


/**The ChainClassifier class provides static
 * functions for computing and reporting the 
 * results of the coreference chain classifier
 * 
 * @author ccervantes
 */
public class ChainClassifier 
{

    /**Predicts coreference chains for the given <b>docSet</b>, using
     * the pairwise scores in <b>scoreDict</b>, where the actual
     * coreference inference procedure is determined by <b>learner</b>.
     * In standard settings (not ILP), multithreading is enabled with
     * <b>numThreads</b>
     *
     * @param docSet        - The documents for which chains will be predicted
     * @param scoreDict_coref     - The pairwise coreference scores
     * @param learner       - Which inference procedure to use
     * @param numThreads    - How many threads to run on (default:1)
     * @param fileRoot      - The root of the output files (default:out)
     * @param exportCoref   - Whether to export the results to a
     *                        <b>fileRoot</b>.coref (default:false)
     * @param exportHtm     - Whether to export a random sample of examples in
     *                        html format (default:false)
     */
    public static Map<String, Set<Chain>>
        predictChains(Collection<Document> docSet,
                      BinaryClassifierScoreDict scoreDict_coref,
                      BinaryClassifierScoreDict scoreDict_nonvis,
                      BinaryClassifierScoreDict scoreDict_null,
                      Map<String, Double> scoreDict_subset,
                      String learner, int numThreads, String fileRoot,
                      boolean exportCoref, boolean exportHtm) {
        //Build our doc/mention dict
        Map<String, Collection<Mention>> docMentionDict = new HashMap<>();
        if (scoreDict_nonvis == null) {
            for (Document d : docSet)
                docMentionDict.put(d.getID(), d.getMentionList());
        } else {
            //if we were given a nonvisual score dict, only store
            //visual mentions
            for (Document d : docSet) {
                docMentionDict.put(d.getID(), new ArrayList<>());
                for (Mention m : d.getMentionList()) {
                    if (scoreDict_nonvis.get(m) != null)
                        if (scoreDict_nonvis.get(m) <= 0)
                            docMentionDict.get(d.getID()).add(m);
                }
            }
        }

        //determine what kind of learner we are, or quit here
        InferenceType infType = null;
        try {
            if (learner != null) {
                learner = learner.toUpperCase().trim();
                infType = InferenceType.valueOf(learner);
            }
        } catch (Exception ex) {/*Do nothing and move on*/}

        //if we had no learner, die
        if (infType == null) {
            Exception ex = new Exception("Invalid learner: " + learner);
            Logger.log(ex);
            System.exit(1);
        }

        //perform pronominal coref and remove predicted pronouns from the
        //set to infer over
        Logger.log("Performing pronominal coreference");
        Map<String, Set<Mention[]>> pronomCorefDict =
                ClassifyUtil.pronominalCoref(docSet);
        int pronomCount_rem = 0;
        for (String docID : pronomCorefDict.keySet()){
            for (Mention[] mentionPair : pronomCorefDict.get(docID)) {
                docMentionDict.get(docID).remove(mentionPair[1]);
                pronomCount_rem++;
            }
        }
        System.out.println("---Removed " + pronomCount_rem + " pronoms -----");

        Logger.log("Associating attributes with agents");
        Map<Mention, AttrStruct> mentionAttrDict =
                ClassifyUtil.attributeAttachment_agent(docSet);
        Map<String, Map<Mention, AttrStruct>> imgMentionAttrDict =
                new HashMap<>();
        for(Mention m : mentionAttrDict.keySet()){
            String docID = m.getDocID();
            if(!imgMentionAttrDict.containsKey(docID))
                imgMentionAttrDict.put(docID, new HashMap<>());
            imgMentionAttrDict.get(docID).put(m, mentionAttrDict.get(m));
        }

        //if we're an ILP solver, solve the ILP
        Logger.log("Predicting chains");
        Map<String, Set<Chain>> docChainSetDict;
        Map<String, List<String>> debugDict = new HashMap<>();
        /*
        if(exportHtm || exportCoref){
            docChainSetDict = infer(docMentionDict, imgMentionAttrDict, scoreDict_coref, infType, numThreads, debugDict);
        } else {
            docChainSetDict = infer(docMentionDict, imgMentionAttrDict, scoreDict_coref, infType, numThreads);
        }*/
        docChainSetDict = infer(docMentionDict, imgMentionAttrDict,
                scoreDict_coref, scoreDict_null, scoreDict_subset,
                infType, numThreads, debugDict);

        //add the pronouns back in to the set of predicted chains
        Logger.log("Attaching pronouns");
        int pronomCount_att = 0;
        for(String docID : pronomCorefDict.keySet()){
            Set<Chain> chainSet = docChainSetDict.get(docID);
            for(Mention[] mPair : pronomCorefDict.get(docID)){
                for(Chain c : chainSet) {
                    if (c.getMentionSet().contains(mPair[0])) {
                        c.addMention(mPair[1]);
                        pronomCount_att++;
                        break;
                    }
                }
            }
        }
        System.out.println("---Added " + pronomCount_att + " pronoms -----");


        Logger.log("Exporting conll files to out/coref/conll/");
        for(Document d : docSet){
            List<String> lineList_key = d.toConll2012();
            lineList_key.add(0, "#begin document (" + d.getID() + "); part 000");
            lineList_key.add("#end document");
            FileIO.writeFile(lineList_key, "out/coref/conll/" + d.getID().replace(".jpg", "") + "_key", "conll", false);

            Set<Chain> predChainSet = docChainSetDict.get(d.getID());
            if(predChainSet == null){
                Logger.log("Document %s has no predicted chains", d.getID());
                predChainSet = new HashSet<>();
            }
            List<String> lineList_resp = Document.toConll2012(d, predChainSet);
            lineList_resp.add(0, "#begin document (" + d.getID() + "); part 000");
            lineList_resp.add("#end document");
            FileIO.writeFile(lineList_resp, "out/coref/conll/" + d.getID().replace(".jpg", "") + "_response", "conll", false);
        }

        Logger.log("Exporting random 100 htm files to out/coref/htm/");
        List<Document> docList = new ArrayList<>(docSet);
        Collections.shuffle(docList);
        for (int i = 0; i < 100; i++) {
            Document d = docList.get(i);
            int prevIdx = i -1;
            if(prevIdx < 0)
                prevIdx = 99;
            int nextIdx = i+1;
            if(nextIdx > 99)
                nextIdx = 0;

            String htm =
                    HtmlScribe.getCorefImgPage(d, docChainSetDict.get(d.getID()),
                            debugDict.get(d.getID()),
                            new Score(), docList.get(prevIdx).getID(),
                            docList.get(nextIdx).getID());
            FileIO.writeFile(htm, "out/coref/htm/" + d.getID(), "htm", false);
        }
        return docChainSetDict;
    }


    /**Predicts coreference chains using <b>type</b> inference, executing
     * on <b>numThreads</b> parallel threads
     *
     * @param scoreDict     - The pairwise coreference scores
     * @param type          - The inference type to use
     * @param numThreads    - The number of threads to use
     * @return              - A mapping of [docID -> [chain] ]
     */
    private static Map<String, Set<Chain>>
                infer(Map<String, Collection<Mention>> docMentionDict,
                      Map<String, Map<Mention, AttrStruct>> imgMentionAttrDict,
                      BinaryClassifierScoreDict scoreDict,
                      InferenceType type, int numThreads)
    {
        return infer(docMentionDict, imgMentionAttrDict, scoreDict, null, null, type, numThreads, null);
    }



    /**Predicts coreference chains using <b>type</b> inference, executing
     * on <b>numThreads</b> parallel threads
     *
     * @param type          - The inference type to use
     * @param numThreads    - The number of threads to use
     * @param debugDict     - A mapping of [docID -> [debugLine] ] to write
     *                        debug output to (needed for later steps)
     *                        (default:null)
     * @return              - A mapping of [docID -> [chain] ]
     */
    private static Map<String, Set<Chain>> infer(Map<String, Collection<Mention>> docMentionDict,
                                                 Map<String, Map<Mention, AttrStruct>> imgMentionAttrDict,
                                                 BinaryClassifierScoreDict scoreDict_coref,
                                                 BinaryClassifierScoreDict scoreDict_null,
                                                 Map<String, Double> scoreDict_subset,
                                                 InferenceType type, int numThreads,
                                                 Map<String, List<String>> debugDict)
    {
        Logger.log("Beginning predicted chain construction");
        int docIdx = 0;

        //create a threadpool of size [whatever we passed as an arg]
        Thread[] threadPool = new Thread[numThreads];

        //create and start our first set of threads, where type
        //sets which threads we create
        List<String> docIdList = new ArrayList<>(docMentionDict.keySet());
        for(int i=0; i<numThreads; i++) {
            String docID = docIdList.get(docIdx);
            switch(type){
                case ILP_BEST_LINK:
                case ILP_ALL_LINK:
                case ILP_ALL_CONSTRAINTS:
                    threadPool[i] = new ILPSolverThread(type, docMentionDict.get(docID), scoreDict_coref);
                    break;
                case ILP_RELATION:
                    threadPool[i] = new ILPSolverThread(type, docMentionDict.get(docID), scoreDict_coref, scoreDict_null, scoreDict_subset);
                    break;
                default:
                    threadPool[i] = new ChainThread(docMentionDict.get(docID), 0.0, type, scoreDict_coref);
                    break;
            }
            docIdx++;
        }
        for(int i=0; i<threadPool.length; i++)
            threadPool[i].start();

        //keep processing threads until theyre all dead and
        //we've gone through all the documents
        Map<String, Set<Chain>> docChainSetDict = new HashMap<>();
        boolean foundLiveThread = true;
        while(docIdx < docIdList.size() || foundLiveThread) {
            foundLiveThread = false;
            for(int i=0; i<numThreads; i++) {
                if(threadPool[i].isAlive()) {
                    foundLiveThread = true;
                } else {
                    //if this is a dead thread, store it
                    if(type.toString().startsWith("ILP")){
                        ILPSolverThread ist = (ILPSolverThread)threadPool[i];
                        docChainSetDict.put(ist.getDocID(), ist.getPredictedChains());
                    } else {
                        ChainThread ct = (ChainThread)threadPool[i];
                        docChainSetDict.put(ct.getDocID(), ct.getPredictedChains());
                        if(debugDict != null)
                            debugDict.put(ct.getDocID(), ct.getDebugOutput());
                    }

                    //independently, if we found a dead thread and we
                    //still have image IDs to iterate through, swap this
                    //dead one out for a live one
                    if(docIdx < docIdList.size()) {
                        Logger.logStatus("Processed %d images (%.2f%%)",
                                docIdx, 100.0*(double)docIdx / docIdList.size());

                        String docID = docIdList.get(docIdx);
                        switch(type){
                            case ILP_BEST_LINK:
                            case ILP_ALL_LINK:
                            case ILP_ALL_CONSTRAINTS:
                                threadPool[i] = new ILPSolverThread(type, docMentionDict.get(docID), scoreDict_coref);
                                break;
                            case ILP_RELATION:
                                threadPool[i] = new ILPSolverThread(type, docMentionDict.get(docID), scoreDict_coref, scoreDict_null, scoreDict_subset);
                                break;
                            default:
                                threadPool[i] = new ChainThread(docMentionDict.get(docID), 0.0, type, scoreDict_coref);
                                break;
                        }
                        threadPool[i].start();
                        foundLiveThread = true;
                        docIdx++;
                    }
                }
            }

            //before we check for threadlife again, let's
            //sleep for a half second so we don't burn
            try{Thread.sleep(500);}
            catch(InterruptedException iEx){/*do nothing*/}
        }

        //go through the thread pool one last time to
        //collect the last of our threads
        for(int i=0; i<numThreads; i++) {
            String docID;
            Set<Chain> predChainSet;
            List<String> debugOutput = null;
            if(type.toString().startsWith("ILP")){
                ILPSolverThread ist = (ILPSolverThread)threadPool[i];
                docID = ist.getDocID();
                predChainSet = ist.getPredictedChains();
            } else {
                ChainThread ct = (ChainThread)threadPool[i];
                docID = ct.getDocID();
                predChainSet = ct.getPredictedChains();
                debugOutput = ct.getDebugOutput();
            }

            //dont bother adding things that are already there
            if(!docChainSetDict.containsKey(docID)){
                docChainSetDict.put(docID, predChainSet);
                if(debugDict != null && debugOutput != null)
                    debugDict.put(docID, debugOutput);
            }
        }
        return docChainSetDict;
    }


    public static void printStats(Map<String, Score> imgScoreDict, Score.ScoreType scoreType)
    {
        System.out.println("---- " + scoreType.toString() + " ----");
        if(scoreType == Score.ScoreType.BLANC){
            Set<Blanc> blancScoreSet = new HashSet<>();
            for(Score s : imgScoreDict.values())
                blancScoreSet.add((Blanc)s);
            Blanc overallScore = Blanc.getAverageBlanc(blancScoreSet);
            System.out.println(overallScore.toLatexString());
        } else if(scoreType == Score.ScoreType.BCUBED) {
            Score[] scoreArr = new Score[imgScoreDict.values().size()];
            scoreArr = imgScoreDict.values().toArray(scoreArr);
            Score overallScore = new Score(scoreArr);
            System.out.println(overallScore.toLatexString());
        }
    }

    public static Map<String, Score>
        getImgScoreDict(Collection<Document> docSet,
                        Map<String, Set<Chain>> imgChainSetDict_pred,
                        Score.ScoreType scoreType)
    {
        Map<String, Score> imgScoreDict = new HashMap<>();
        for(Document d : docSet){
            Score s = null;
            Set<Chain> goldChainSet = new HashSet<>();
            for(Chain c : d.getChainSet()){
                if(!c.getID().equals("0")){
                    goldChainSet.add(c);
                } else {
                    int nonvisCounter = 0;
                    for(Mention m : c.getMentionSet()){
                        Chain cPrime = new Chain(d.getID(), "0_" +nonvisCounter);
                        cPrime.addMention(m);
                        goldChainSet.add(cPrime);
                        nonvisCounter++;
                    }
                }
            }

            switch(scoreType){
                case BCUBED: s = new BCubed(goldChainSet,imgChainSetDict_pred.get(d.getID()));
                    break;
                case BLANC: s = new Blanc(goldChainSet, imgChainSetDict_pred.get(d.getID()));
                    break;
                /*
                case BCUBED: s = new BCubed(d.getChainSet(),imgChainSetDict_pred.get(d.getID()));
                    break;
                case BLANC: s = new Blanc(d.getChainSet(), imgChainSetDict_pred.get(d.getID()));
                    break;
            */
            }
            if(s != null)
                imgScoreDict.put(d.getID(), s);
        }
        return imgScoreDict;
    }

    public enum InferenceType{
        GREEDY, SEQUENCE_ASC, SEQUENCE_DESC, SEQUENCE_RAND,
        TE_PREMISE_ONLY, TE_FULL, ILP_BEST_LINK, ILP_ALL_LINK, ILP_ALL_CONSTRAINTS,
        ILP_RELATION
    }
}
