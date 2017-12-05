package learn;

import core.Minion;
import core.Overlord;
import org.apache.commons.lang.ArrayUtils;
import out.OutTable;
import statistical.ScoreDict;
import structures.*;
import utilities.*;

import java.io.File;
import java.util.*;

import static learn.ILPInference.InferenceType.RELATION;
import static learn.ILPInference.InferenceType.VISUAL_RELATION;

/**The ILPInference class provides static
 * functions for computing and reporting the 
 * results of the coreference chain classifier
 * 
 * @author ccervantes
 */
public class ILPInference
{
    private Map<String, Map<String, Integer>> _relationGraphs, _groundingGraphs, _visualGraphs;
    private Map<String, Document> _docDict;
    private Map<String, Set<Chain>> _predChains;
    private InferenceType _infType;
    private Map<String, List<BoundingBox>> _boxDict;
    private Map<String, double[]> _relationScores, _affinityScores, _cardinalityScores;
    private Map<String, Double> _nonvisScores;
    private Set<String> _failedImgs, _fallbackImgs;
    private String _graphRoot;
    private DoubleDict<String> _groundingAccuracies, _relationAccuracies;
    private boolean _includeTypeConstr, _excludeBoxExigence, _excludeSubset, _onlyKeepPositiveLinks;

    /**Creates a new ILPInference module, using the specified
     * docSet and nonvisual scores file; Performs combined inference
     * over given scores
     *
     * @param docSet
     * @param nonvisScoresFile
     * @param relationScoresFile
     * @param affinityScoresFile
     * @param cardinalityScoresFile
     * @param graphRoot
     * @param includeTypeConstr
     * @param excludeBoxExigence
     * @param excludeSubset
     * @param onlyKeepPositiveLinks
     */
    public ILPInference(Collection<Document> docSet, InferenceType infType,
                        String nonvisScoresFile, String relationScoresFile,
                        String affinityScoresFile, String cardinalityScoresFile,
                        String graphRoot, boolean includeTypeConstr,
                        boolean excludeBoxExigence, boolean excludeSubset,
                        boolean onlyKeepPositiveLinks)
    {
        //If we're going to use the type constraint, intitialize the lexicons
        _includeTypeConstr = includeTypeConstr;
        if(_includeTypeConstr)
            Mention.initializeLexicons(Overlord.flickr30k_lexicon, Overlord.mscoco_lexicon);

        //Store the simple vars
        _onlyKeepPositiveLinks = onlyKeepPositiveLinks;
        _excludeBoxExigence = excludeBoxExigence;
        _excludeSubset = excludeSubset;
        _infType = infType;

        //If specified, load the last graph attempts
        _relationGraphs = new HashMap<>();
        _groundingGraphs = new HashMap<>();
        _visualGraphs = new HashMap<>();
        _graphRoot = graphRoot;
        if(_graphRoot != null)
            _loadGraphs();

        //initialize everything
        _groundingAccuracies = new DoubleDict<>(); _relationAccuracies = new DoubleDict<>();
        _predChains = new HashMap<>(); _failedImgs = new HashSet<>(); _fallbackImgs = new HashSet<>();

        //map the documents to their IDs
        _docDict = new HashMap<>();
        for(Document d : docSet)
            _docDict.put(d.getID(), d);

        //Load and evaluate those scores files
        _loadAndEvalScores(relationScoresFile, affinityScoresFile,
                cardinalityScoresFile, nonvisScoresFile);

        //store our bounding boxes unless this is a
        //relation inference object
        _boxDict = new HashMap<>();
        if(_infType != InferenceType.RELATION){
            for(Document d : _docDict.values()){
                List<BoundingBox> boxes = new ArrayList<>(d.getBoundingBoxSet());
                if(boxes.isEmpty())
                    Logger.log("WARNING: Image " + d.getID() + " has no boxes");
                _boxDict.put(d.getID(), boxes);
            }
        }
    }

    /* File loading methods */

    /**Loads the specified scores files into internal
     * dictionaries, evaluating each as an easy point
     * of comparison in the log file
     *
     * @param relationScoresFile
     * @param affinityScoresFile
     * @param cardScoresFile
     */
    private void _loadAndEvalScores(String relationScoresFile,
            String affinityScoresFile, String cardScoresFile, String nonvisScoresFile)
    {
        _relationScores = new HashMap<>();
        _affinityScores = new HashMap<>();
        _cardinalityScores = new HashMap<>();
        _nonvisScores = new HashMap<>();

        /* Load the appropriate scores; bail if there's a mismatch */
        Logger.log("Reading scores file(s)");
        if(relationScoresFile != null){
            _relationScores = ClassifyUtil.readMccScoresFile(relationScoresFile);
        }
        if(affinityScoresFile != null && cardScoresFile != null){
            _affinityScores = ClassifyUtil.readMccScoresFile(affinityScoresFile);
            _cardinalityScores = ClassifyUtil.readMccScoresFile(cardScoresFile);
        }
        Set<String> freqNonvisHeads = new HashSet<>();
        if(nonvisScoresFile != null){
            //If we're going to predict nonvis mentions, load the head word file
            //for a baseline heuristic comparison
            freqNonvisHeads =
                    ClassifyUtil.loadOnehotDict(Overlord.flickr30kResources +
                            "hist_nonvisual.csv", 10).keySet();

            //Store the nonvis scores only (since this is a binary classifier,
            //we can just do 1-nonvis to get the vis score)
            Map<String, double[]> nonvisScoreDict =
                    ClassifyUtil.readMccScoresFile(nonvisScoresFile);
            for(String id : nonvisScoreDict.keySet())
                _nonvisScores.put(id, nonvisScoreDict.get(id)[1]);
        }

        /* To easily compare to all downstream tasks, print scores as
         * we read them from their files
         */
        Map<String, Integer> predLabelDict_vis = new HashMap<>();
        if(!_nonvisScores.isEmpty()){
            Logger.log("---------- Visual Scores ----------");
            //NOTE: these predicted label dicts should store _visual_ as 1

            //Log our heuristic results, as a baseline
            Map<String, Integer> predLabelDict_heur = new HashMap<>();
            for(Document d : _docDict.values()){
                for(Mention m : d.getMentionList()){
                    int pred = 1;
                    if(freqNonvisHeads.contains(m.getHead().toString().toLowerCase()))
                        pred = 0;
                    predLabelDict_heur.put(m.getUniqueID(), pred);
                }
            }
            _evaluateVisual(predLabelDict_heur, "heuristic");

            //Log the results of our classifier
            _nonvisScores.forEach((k,v) -> predLabelDict_vis.put(k, v < 0.5 ? 1 : 0));

            //Add any mentions that we don't have labels for as predicted visual
            //(the dominant class)
            Set<Mention> unknownMentions = new HashSet<>();
            for(Document d : _docDict.values())
                for(Mention m : d.getMentionList())
                    if(!predLabelDict_vis.containsKey(m.getUniqueID()))
                        unknownMentions.add(m);
            Logger.log("WARNING: Found " + unknownMentions.size() + " mentions without predicted nonvis labels; assigning as visual");
            unknownMentions.forEach(m -> predLabelDict_vis.put(m.getUniqueID(), 1));

            _evaluateVisual(predLabelDict_vis, "pairwise");
        }

        if(!_relationScores.isEmpty()){
            Logger.log("---------- Relation Scores ----------");
            Map<String, Integer> predLabelDict = new HashMap<>();
            _relationScores.forEach((k,v) ->
                    predLabelDict.put(k, Util.getMaxIdx(ArrayUtils.toObject(v))));
            _evaluateRelations(predLabelDict, "out/pre_inf_rel");
            Map<String, Set<Chain>> predChains = _buildChainsFromPredLabels(predLabelDict, predLabelDict_vis);
            for(Document d : _docDict.values())
                _exportConllFile(d, predChains.get(d.getID()), "pairwise");
        }
        if(!_affinityScores.isEmpty()){
            Logger.log("---------- Affinity Scores ----------");
            Map<String, Integer> predLabelDict = new HashMap<>();
            _affinityScores.forEach((k,v) ->
                    predLabelDict.put(k, Util.getMaxIdx(ArrayUtils.toObject(v))));
            _evaluateGroundings(predLabelDict, "out/pre_inf_affinity");
        }
        if(!_cardinalityScores.isEmpty()){
            Logger.log("---------- Cardinality Scores ----------");
            Map<String, Integer> predLabelDict = new HashMap<>();
            _cardinalityScores.forEach((k,v) ->
                    predLabelDict.put(k, Util.getMaxIdx(ArrayUtils.toObject(v))));
            ScoreDict<Integer> scores = _evaluateCardinality(predLabelDict);
            scores.printCompleteScores();
            System.out.printf("RMSE: %.2f\n", scores.getRMSE());
        }
    }

    /**Loads the previous attempt's solved graphs so we can make
     * incremental progress
     */
    private void _loadGraphs()
    {
        String filename_relation = _graphRoot + "_REL.obj";
        String filename_grounding = _graphRoot + "_GRND.obj";
        String filename_visual = _graphRoot + "_VIS.obj";
        File f_rel = new File(filename_relation);
        File f_grnd = new File(filename_grounding);
        File f_vis = new File(filename_visual);

        if(f_rel.exists())
            _relationGraphs = (Map<String, Map<String, Integer>>)
                    FileIO.readObject(Map.class, filename_relation);
        if(f_grnd.exists())
            _groundingGraphs = (Map<String, Map<String, Integer>>)
                    FileIO.readObject(Map.class, filename_grounding);
        if(f_vis.exists())
            _visualGraphs = (Map<String, Map<String, Integer>>)
                    FileIO.readObject(Map.class, filename_visual);

        if(_relationGraphs == null)
            _relationGraphs = new HashMap<>();
        if(_groundingGraphs == null)
            _groundingGraphs = new HashMap<>();
        if(_visualGraphs == null)
            _visualGraphs = new HashMap<>();

        Logger.log("Loaded %d relation graphs; %d grounding graphs; %d nonvis graphs",
                   _relationGraphs.size(), _groundingGraphs.size(), _visualGraphs.size());
    }

    /* Evaluation methods */

    private void _printDocumentScoreDict(Map<Document, ScoreDict<String>> docScoreDict, String filename)
    {
        //store the micro-averaged accuracies and the macro-averaged scores by bin
        Map<Integer, List<Double>> accuracies_byEntities = new HashMap<>();
        Map<Integer, ScoreDict<String>> macroAverage_byEntities = new HashMap<>();
        ScoreDict<String> macroAverage = new ScoreDict<>();
        Set<String> labelSet = new HashSet<>();
        List<Double> accuracies = new ArrayList<>();
        for(Document d : docScoreDict.keySet()){
            int numEntities = d.getChainSet().size();
            ScoreDict<String> scores = docScoreDict.get(d);
            macroAverage.increment(scores);
            accuracies.add(scores.getAccuracy());
            labelSet.addAll(scores.keySet());

            //bin the accuracy and macro average link counts by entity count
            if(!accuracies_byEntities.containsKey(numEntities)) {
                accuracies_byEntities.put(numEntities, new ArrayList<>());
                macroAverage_byEntities.put(numEntities, new ScoreDict<>());
            }
            accuracies_byEntities.get(numEntities).add(scores.getAccuracy());
            macroAverage_byEntities.get(numEntities).increment(scores);
        }
        List<String> labels = new ArrayList<>(labelSet);

        Logger.log("---- Printing macro-averaged scores ----");
        macroAverage.printCompleteScores();

        List<Integer> entityCounts = new ArrayList<>(accuracies_byEntities.keySet());
        Collections.sort(entityCounts);
        //set up our header values
        List<String> headers = new ArrayList<>();
        headers.add("entities"); headers.add("doc_freq");
        headers.add("micro_acc"); headers.add("perfect");
        headers.add("macro_acc");
        for(String y : labels){
            headers.add("P_" + y); headers.add("R_" + y);
            headers.add("F1_" + y); headers.add("gold_links_" + y);
        }
        headers.add("total_links");

        //Store these as the columns of an OutTable
        OutTable ot = new OutTable(headers.toArray(new String[headers.size()]));
        int perfects_total = 0;
        for(Integer entityCount : entityCounts){
            List<Object> row = new ArrayList<>();
            List<Double> acc_byEntityBin = accuracies_byEntities.get(entityCount);
            ScoreDict<String> scores = macroAverage_byEntities.get(entityCount);

            row.add(entityCount);
            row.add(acc_byEntityBin.size());
            row.add(StatisticalUtil.getMean(acc_byEntityBin) / 100.0);
            int perfects = 0;
            for(Double acc : acc_byEntityBin)
                if(acc == 100)
                    perfects++;
            perfects_total += perfects;
            row.add(perfects);
            row.add(scores.getAccuracy());
            for(String label : labels){
                row.add(scores.getScore(label).getPrecision());
                row.add(scores.getScore(label).getRecall());
                row.add(scores.getScore(label).getF1());
                row.add(scores.getGoldCount(label));
            }
            row.add(scores.getTotalGold());
            ot.addRow(row.toArray(new Object[row.size()]));
        }
        if(filename != null){
            Logger.log("---- Writing per/entity scores to "+ filename + " ----");
            ot.writeToCsv(filename);
        }
        System.out.printf("Average Image Accuracy: %.2f%%\n",
                StatisticalUtil.getMean(accuracies));
        System.out.printf("Perfects: %d (%.2f%%)\n", perfects_total,
                100.0 * (double)perfects_total / (double)_docDict.size());
    }

    private void _evaluateGroundings(Map<String, Integer> predLabelDict, String filename)
    {
        Map<Document, ScoreDict<String>> scoreDict = new HashMap<>();
        DoubleDict<Document> groundingDict = new DoubleDict<>();
        Map<String, ScoreDict<String>> scoreDict_byCat = new HashMap<>();
        DoubleDict<String> catDistro = new DoubleDict<>();
        for (Document d : _docDict.values()) {
            List<Mention> mentions = d.getMentionList();
            Set<BoundingBox> boxes = d.getBoundingBoxSet();
            ScoreDict<String> scores = new ScoreDict<>();
            for (Mention m : mentions) {
                Set<BoundingBox> assocBoxes = d.getBoxSetForMention(m);
                boolean nonvisMention = m.getChainID().equals("0");

                //if (_usePredictedNonvis)
                //    nonvisMention = _nonvisMentions.contains(m.getUniqueID());

                boolean foundConflictingLink = false;
                for (BoundingBox b : boxes) {
                    int gold = assocBoxes.contains(b) ? 1 : 0;

                    //if this is a nonvisual mention according to
                    //our scheme, it's always pred 0
                    int pred = 0;
                    if (!nonvisMention) {
                        String id = m.getUniqueID() + "|" + b.getUniqueID();
                        if (predLabelDict.containsKey(id))
                            pred = predLabelDict.get(id);
                    }
                    scores.increment(String.valueOf(gold), String.valueOf(pred));
                    if(gold != pred)
                        foundConflictingLink = true;


                    if(!scoreDict_byCat.containsKey(b.getCategory()))
                        scoreDict_byCat.put(b.getCategory(), new ScoreDict<>());
                    scoreDict_byCat.get(b.getCategory()).increment(String.valueOf(gold),
                            String.valueOf(pred));
                    catDistro.increment(b.getCategory());
                    catDistro.increment("total");
                }

                if(!foundConflictingLink)
                    groundingDict.increment(d);
            }
            scoreDict.put(d, scores);
        }
        _printDocumentScoreDict(scoreDict, filename);


        /*
        System.out.println("-----------------");
        for(String s : scoreDict_byCat.keySet()) {
            Score sc = scoreDict_byCat.get(s).getScore("1");
            System.out.printf("%s,%d,%.2f,%.2f,%.2f,%.2f\n", s,
                    (int)catDistro.get(s), catDistro.get(s)/catDistro.get("total"),
                    sc.getPrecision(), sc.getRecall(), sc.getF1());
        }
        System.out.println("-----------------");
        */


        //Store these image accuracies; otherwise store their difference
        //and print the 50 most improved images
        if(_groundingAccuracies.size() == 0){
            for(Document d : scoreDict.keySet())
                _groundingAccuracies.increment(d.getID(), scoreDict.get(d).getAccuracy());
        } else {
            for(Document d : scoreDict.keySet()){
                double acc = _groundingAccuracies.get(d.getID());
                _groundingAccuracies.remove(d.getID());
                _groundingAccuracies.increment(d.getID(), scoreDict.get(d).getAccuracy() - acc);
            }

            int i=0;
            Logger.log("Images with greatest accuracy diff between pre and post inference");
            for(String docID : _groundingAccuracies.getSortedByValKeys(true)) {
                System.out.printf("%s: %.2f%% (+%.2f%%)\n", docID,
                        scoreDict.get(_docDict.get(docID)).getAccuracy(),
                        _groundingAccuracies.get(docID));
                i++;
                if(i >= 50)
                    break;
            }
        }

        /* We also want to evaluate how often we get groundings perfectly correct */
        double perfectMentions = groundingDict.getSum();
        double totalMentions = 0;
        for(Document d : groundingDict.keySet())
            totalMentions += d.getMentionList().size();

        System.out.printf("Found %d (%.2f%%) correctly grounded mentions\n",
                (int)perfectMentions, 100.0 * perfectMentions / totalMentions);
    }

    private void _evaluateRelations(Map<String, Integer> predLabelDict, String filename)
    {
        Map<Document, ScoreDict<String>> docScoreDict = new HashMap<>();
        Map<Document, ScoreDict<String>> docScoreDict_intra = new HashMap<>();
        Map<Document, ScoreDict<String>> docScoreDict_inter = new HashMap<>();
        for(Document d : _docDict.values()) {
            ScoreDict<String> scores = new ScoreDict<>();
            ScoreDict<String> scores_intra = new ScoreDict<>(), scores_inter = new ScoreDict<>();
            Set<String> subsetMentions = d.getSubsetMentions();
            List<Mention> mentionList = d.getMentionList();

            for (int i = 0; i < mentionList.size(); i++) {
                Mention m_i = mentionList.get(i);
                for (int j = i + 1; j < mentionList.size(); j++) {
                    Mention m_j = mentionList.get(j);

                    String id_ij = Document.getMentionPairStr(m_i, m_j);
                    String id_ji = Document.getMentionPairStr(m_j, m_i);
                    boolean nonvisMention = m_i.getChainID().equals("0") || m_j.getChainID().equals("0");

                    String gold = "null";
                    if(!nonvisMention){
                        if(m_i.getChainID().equals(m_j.getChainID())){
                            gold = "coref";
                        } else if(subsetMentions.contains(id_ij)) {
                            gold = "subset_ij";
                        } else if(subsetMentions.contains(id_ji)){
                            gold = "subset_ji";
                        }
                    }

                    String pred = "-invalid-";
                    if(predLabelDict.containsKey(id_ij) && predLabelDict.containsKey(id_ji)){
                        int pred_ij = predLabelDict.get(id_ij);
                        int pred_ji = predLabelDict.get(id_ji);
                        if(pred_ij == pred_ji && pred_ij == 0){
                            pred = "null";
                        } else if(pred_ij == pred_ji && pred_ij == 1){
                            pred = "coref";
                        }
                        else if(pred_ij == 2 && pred_ji == 3 ||
                                  pred_ij == 3 && pred_ji == 2){
                            if(pred_ij == 2)
                                pred = "subset_ij";
                            else
                                pred = "subset_ji";
                        }
                    }

                    //Handle subset pairs according to whether the direction
                    //is correct
                    if(gold.startsWith("subset_") && pred.startsWith("subset_")){
                        //If both links are subset and their direction matches
                        //drop the direction (since they're a match)
                        if(gold.equals(pred)){
                            gold = "subset"; pred = "subset";
                        }
                        //If both links are subset and their direction does
                        //_not_ match, drop gold's direction and drop pred's
                        //label entirely (since they're not a match)
                        else {
                            gold = "subset"; pred = "-reverse_sub-";
                        }
                    }

                    //In all other subset cases, drop the direction
                    if(gold.startsWith("subset_"))
                        gold = "subset";
                    if(pred.startsWith("subset_"))
                        pred = "subset";

                    //Increment the appropriate scores, skipping gold nonvisual pairs
                    scores.increment(gold, pred);
                    if(m_i.getCaptionIdx() == m_j.getCaptionIdx())
                        scores_intra.increment(gold, pred);
                    else
                        scores_inter.increment(gold, pred);
                }
            }
            docScoreDict.put(d, scores);
            docScoreDict_intra.put(d, scores_intra);
            docScoreDict_inter.put(d, scores_inter);
        }

        Logger.log("----- Overall Scores ------");
        _printDocumentScoreDict(docScoreDict, filename);
        Logger.log("----- Intra-Caption ------");
        _printDocumentScoreDict(docScoreDict_intra, null);
        Logger.log("----- Cross-Caption ------");
        _printDocumentScoreDict(docScoreDict_inter, null);


        //Store these image accuracies; otherwise store their difference
        //and print the 50 most improved images
        if(filename.contains("pronom")) //reset on the pronom stage, since we call this method three times
            _relationAccuracies = new DoubleDict<>();

        if(_relationAccuracies.size() == 0){
            for(Document d : docScoreDict.keySet())
                _relationAccuracies.increment(d.getID(), docScoreDict.get(d).getAccuracy());
        } else {
            for(Document d : docScoreDict.keySet()){
                double acc = _relationAccuracies.get(d.getID());
                _relationAccuracies.remove(d.getID());
                _relationAccuracies.increment(d.getID(), docScoreDict.get(d).getAccuracy() - acc);
            }

            int i=0;
            Logger.log("Images with greatest accuracy diff between pre and post inference");
            for(String docID : _relationAccuracies.getSortedByValKeys(true)) {
                System.out.printf("%s: %.2f%% (+%.2f%%)\n", docID,
                        docScoreDict.get(_docDict.get(docID)).getAccuracy(),
                        _relationAccuracies.get(docID));
                i++;
                if(i >= 50)
                    break;
            }
        }

        //Additionally, let's evaluate to what the percentage of perfect
        //entities have been predicted
        Map<Document, Integer> correctEntityCounts = new HashMap<>();
        for(Document d : _docDict.values()){
            int numEntitiesFound = 0;
            List<Mention> mentions = d.getMentionList();
            Set<String> subsetMentions = d.getSubsetMentions();

            //An entity is said to be correct when all links to/from all
            //mentions within that entity are correct
            for(Chain c : d.getChainSet()){
                boolean foundConflict = false;
                for(Mention m_i : c.getMentionSet()){
                    for(Mention m_j : mentions){
                        if(m_i.equals(m_j))
                            continue;

                        String id_ij = Document.getMentionPairStr(m_i, m_j);
                        String id_ji = Document.getMentionPairStr(m_j, m_i);

                        boolean nonvis_j = m_j.getChainID().equals("0");

                        int gold = 0;
                        if(!nonvis_j){
                            if(m_i.getChainID().equals(m_j.getChainID()))
                                gold = 1;
                            else if(subsetMentions.contains(id_ij))
                                gold = 2;
                            else if(subsetMentions.contains(id_ji))
                                gold = 3;
                        }

                        int pred = -1;
                        if(predLabelDict.containsKey(id_ij) && predLabelDict.containsKey(id_ji)){
                            int pred_ij = predLabelDict.get(id_ij);
                            int pred_ji = predLabelDict.get(id_ji);
                            if(pred_ij == pred_ji && pred_ij == 0){
                                pred = 0;
                            } else if(pred_ij == pred_ji && pred_ij == 1){
                                pred = 1;
                            } else if(pred_ij == 2 && pred_ji == 3 &&
                                      pred_ij == 3 && pred_ji == 2){
                                if(pred_ij == 2)
                                    pred = 2;
                                else
                                    pred = 3;
                            }
                        }

                        foundConflict |= gold != pred;
                    }
                }

                if(!foundConflict)
                    numEntitiesFound++;
            }

            correctEntityCounts.put(d, numEntitiesFound);
        }

        double correctEntities = 0.0, totalEntities = 0.0;
        DoubleDict<String> entityAverages = new DoubleDict<>();
        for(Document d : correctEntityCounts.keySet()){
            correctEntities += correctEntityCounts.get(d);
            totalEntities += d.getChainSet().size();

            double microAverage = (double)correctEntityCounts.get(d) / (double)d.getChainSet().size();
            entityAverages.increment(d.getID(), microAverage);
        }

        System.out.printf("Found %d (%.2f%%) correct entities\n",
                (int)correctEntities, 100.0 * correctEntities / totalEntities);
    }

    private void _evaluateVisual(Map<String, Integer> predLabelDict, String predType)
    {
        Map<Document, ScoreDict<String>> scoreDict = new HashMap<>();
        for(Document d : _docDict.values()){
            ScoreDict<String> scores = new ScoreDict<>();
            for(Mention m : d.getMentionList()){
                int gold = m.getChainID().equals("0") ? 0 : 1;
                int pred = 1;
                if(predLabelDict.containsKey(m.getUniqueID()))
                    pred = predLabelDict.get(m.getUniqueID());
                scores.increment(String.valueOf(gold), String.valueOf(pred));
            }
            scoreDict.put(d, scores);
        }
        System.out.println(predType);
        _printDocumentScoreDict(scoreDict, null);
    }

    /**Evaluates the predicted box cardinalities in the predLabelDict
     *
     * @param predLabelDict
     * @return
     */
    private ScoreDict<Integer> _evaluateCardinality(Map<String, Integer> predLabelDict)
    {
        ScoreDict<Integer> scores = new ScoreDict<>();
        for(Document d : _docDict.values()){
            List<Mention> mentions = d.getMentionList();
            for(Mention m : mentions){
                //get the boxes for this mention and whether
                //this is a nonvis mention (according to our scheme)
                Set<BoundingBox> assocBoxes = d.getBoxSetForMention(m);
                boolean nonvisMention = m.getChainID().equals("0");

                //Treat all >10 boxes equally
                int gold = Math.min(assocBoxes.size(), 11);

                //predicted nonvisual mentions are always pred 0
                int pred = 0;
                if(!nonvisMention)
                    if(predLabelDict.containsKey(m.getUniqueID()))
                        pred = predLabelDict.get(m.getUniqueID());

                scores.increment(gold, pred);
            }
        }
        return scores;
    }

    /**Performs post-inference evaluation, including writing
     * conll and htm output files, if specified
     *
     * @param exportFiles
     */
    public void evaluate(boolean exportFiles)
    {
        if(InferenceType.isVisualType(_infType)){
            Logger.log("Evaluating visual graphs");
            Map<String, Integer> predLabels_vis = new HashMap<>();
            for(Map<String, Integer> labelDict : _visualGraphs.values())
                labelDict.forEach((k,v) -> predLabels_vis.put(k, v));
            _evaluateVisual(predLabels_vis, "inf");
        }
        if(InferenceType.isRelationType(_infType)){
            Logger.log("Evaluating relation graphs");
            Map<String, Integer> predLabels = new HashMap<>();
            for(Map<String, Integer> labelDict : _relationGraphs.values())
                labelDict.forEach((k,v) -> predLabels.put(k, v));
            _evaluateRelations(predLabels, "out/post_inf_rel");
        }
        if(InferenceType.isGroundingType(_infType)){
            Logger.log("Evaluating grounding graphs");
            Map<String, Integer> predLabels = new HashMap<>();
            for(Map<String, Integer> labelDict : _groundingGraphs.values())
                labelDict.forEach((k,v) -> predLabels.put(k, v));
            _evaluateGroundings(predLabels, "out/post_inf_ground");
        }

        //Export files, if specified
        if(exportFiles){
            if(InferenceType.isRelationType(_infType)){
                Logger.log("Exporting conll and latex files");
                Map<String, Set<Chain>> docPredChains = getPredictedChains();
                for(Document d : _docDict.values()){
                    String corefCase = _infType == InferenceType.RELATION ? "inf" : "joint";
                    _exportConllFile(d, docPredChains.get(d.getID()), corefCase);
                    _exportRelationFile(d, docPredChains.get(d.getID()));
                }

                Logger.log("Exporting htm files");
                Map<String, Set<Chain[]>> predSubsetChains = getPredictedSubsetChains();
                for(Document d : _docDict.values()){
                    FileIO.writeFile(HtmlIO.getImgHtm(d, docPredChains.get(d.getID()),
                            predSubsetChains.get(d.getID())),
                            "out/" + _infType.toString().toLowerCase() + "/htm/" +
                            d.getID().replace(".jpg", ""), "htm", false);
                }
            }
            if(InferenceType.isGroundingType(_infType)){
                Logger.log("Exporting grounding files");
                Map<String, Integer> predLabels = new HashMap<>();
                for(Map<String, Integer> labelDict : _groundingGraphs.values())
                    labelDict.forEach((k,v) -> predLabels.put(k, v));
                for(Document d : _docDict.values())
                    _exportGroundingFile(d, predLabels);
            }
        }
    }

    /**Exports a CONLL file given a document and predicted chain set;
     * the exported file is placed in a different directory, depending on
     * the given coref case
     *
     * @param d
     * @param predChainSet
     * @param corefCase     - corefCase \in {pairwise, plus_pronom, inf, joint}
     */
    private void _exportConllFile(Document d, Set<Chain> predChainSet, String corefCase)
    {
        String outDir = "out/";
        switch(corefCase){
            case "pairwise": outDir += "relation/" + corefCase + "_conll/";
                break;
            case "plus_pronom": outDir += "relation/" + corefCase + "_conll/";
                break;
            case "inf": outDir += "relation/" + corefCase + "_conll/";
                break;
            case "joint": outDir += corefCase + "/conll/";
                break;
        }

        //Write the key file
        List<String> lineList_key = d.toConll2012();
        lineList_key.add(0, "#begin document (" + d.getID() + "); part 000");
        lineList_key.add("#end document");
        FileIO.writeFile(lineList_key, outDir +
                d.getID().replace(".jpg", "") + "_key", "conll", false);

        //Write the response file
        Set<Chain> predChainSet_conll = new HashSet<>();
        if(predChainSet == null) {
            Logger.log("Document %s has no predicted chains", d.getID());
        } else {
            //if we have a chain set, remove chain 0 for the conll file
            for(Chain c : predChainSet)
                if(!c.getID().equals("0"))
                    predChainSet_conll.add(c);
        }
        List<String> lineList_resp = Document.toConll2012(d, predChainSet_conll);
        lineList_resp.add(0, "#begin document (" + d.getID() + "); part 000");
        lineList_resp.add("#end document");
        FileIO.writeFile(lineList_resp, outDir +
                d.getID().replace(".jpg", "") + "_response", "conll", false);
    }

    private void _exportRelationFile(Document d, Set<Chain> predChains)
    {
        //Get chain colors for gold and predicted chains
        List<String> chainIds_gold = new ArrayList<>(), chainIds_pred = new ArrayList<>();
        d.getChainSet().forEach(c -> chainIds_gold.add(c.getID()));
        predChains.forEach(c -> chainIds_pred.add(c.getID()));
        Map<String, String> chainColors_gold = Minion.getLatexColors(chainIds_gold);
        Map<String, String> chainColors_pred = Minion.getLatexColors(chainIds_pred);

        //Get a mapping of mentions to predicted chain IDs
        Map<String, String> predMentionChainDict = new HashMap<>();
        for(Chain c : predChains)
            for(Mention m : c.getMentionSet())
                predMentionChainDict.put(m.getUniqueID(), c.getID());

        List<String> ll_gold = new ArrayList<>(), ll_pred = new ArrayList<>();
        for(Caption c : d.getCaptionList()){
            Map<String, String> mentionChainDict_gold = new HashMap<>();
            Map<String, String> mentionChainDict_pred = new HashMap<>();
            for(Mention m : c.getMentionList()){
                boolean nonvisMention = m.getChainID().equals("0");
                if(!nonvisMention)
                    mentionChainDict_gold.put(m.getUniqueID(), m.getChainID());

                if(!nonvisMention)
                    mentionChainDict_pred.put(m.getUniqueID(), predMentionChainDict.get(m.getUniqueID()));
            }
            ll_gold.add(c.toLatexString(mentionChainDict_gold, chainColors_gold, true, true) + "\\\\");
            ll_pred.add(c.toLatexString(mentionChainDict_pred, chainColors_pred, true, true) + "\\\\");
        }

        String outDir = "out/relation/";
        List<String> ll = new ArrayList<>();
        ll.add("\\emph{Gold}\\\\");
        ll.addAll(ll_gold);
        ll.add("\\emph{Predicted}\\\\");
        ll.addAll(ll_pred);
        FileIO.writeFile(ll, outDir + d.getID().replace(".jpg", ""), "txt", false);
    }

    /**Export a grounding file for a given document and predicted label dict;
     * the exported file is placed in a different directory, depending on
     * the inference type
     *
     * @param d
     * @param groundingPredLabelDict
     */
    private void _exportGroundingFile(Document d, Map<String, Integer> groundingPredLabelDict)
    {
        List<BoundingBox> boxes = new ArrayList<>(d.getBoundingBoxSet());

        //Colors may be too distracting for bounding boxes
        //boxes.forEach(b -> boxIds.add("b-"+b.getIdx()));
        //Map<String, String> boxColors = Minion.getLatexColors(boxIds);

        List<String> ll_gold = new ArrayList<>(), ll_pred = new ArrayList<>();
        for(Caption c : d.getCaptionList()) {
            Map<String, String> mentionBoxDict_gold = new HashMap<>();
            Map<String, String> mentionBoxDict_pred = new HashMap<>();
            for (Mention m : c.getMentionList()) {
                //Get the gold mapping of associated boxes and put them under
                //this mention's entry
                Set<BoundingBox> assocBoxes = d.getBoxSetForMention(m);
                List<String> assocBoxIds_gold = new ArrayList<>();
                assocBoxes.forEach(b -> assocBoxIds_gold.add("b-" + b.getIdx()));
                Collections.sort(assocBoxIds_gold);
                mentionBoxDict_gold.put(m.getUniqueID(), StringUtil.listToString(assocBoxIds_gold, ", "));

                //get the predicted mapping of assicated boxes and put them
                //under this mention's entry
                List<String> assocBoxIds_pred = new ArrayList<>();
                boolean nonvisMention = m.getChainID().equals("0");
                /*
                if (_usePredictedNonvis)
                    nonvisMention = _nonvisMentions.contains(m.getUniqueID());*/
                if (!nonvisMention) {
                    for (int o = 0; o < boxes.size(); o++) {
                        BoundingBox b = boxes.get(o);
                        String id = m.getUniqueID() + "|" + b.getUniqueID();
                        if (groundingPredLabelDict.containsKey(id) && groundingPredLabelDict.get(id) == 1)
                            assocBoxIds_pred.add("b-" + b.getIdx());
                    }
                }
                mentionBoxDict_pred.put(m.getUniqueID(), StringUtil.listToString(assocBoxIds_pred, ", "));
            }
            ll_gold.add(c.toLatexString(mentionBoxDict_gold) + "\\\\");
            ll_pred.add(c.toLatexString(mentionBoxDict_pred) + "\\\\");
        }

        String outDir = "out/grounding/";
        List<String> ll = new ArrayList<>();
        ll.add("\\emph{Gold}\\\\");
        ll.addAll(ll_gold);
        ll.add("\\emph{Predicted}\\\\");
        ll.addAll(ll_pred);
        FileIO.writeFile(ll, outDir + d.getID().replace(".jpg", ""), "txt", false);
    }

    /* Graph conversion functions */

    /**Builds predicted chains from a given label dict
     */
    private Map<String, Set<Chain>> _buildChainsFromPredLabels(Map<String, Integer> predLabelDict_rel,
                                                               Map<String, Integer> predLabelDict_vis)
    {
        Map<String, Set<Chain>> predChains = new HashMap<>();
        for(Document d : _docDict.values()){
            Map<Mention, String> mentionChainIdDict = new HashMap<>();
            int chainIdx = 1;
            List<Mention> mentions = new ArrayList<>(d.getMentionList());
            for(int i=0; i<mentions.size(); i++){
                Mention m_i = mentions.get(i);

                //Skip all predicted nonvisual mentions
                if(predLabelDict_vis.containsKey(m_i.getUniqueID()) &&
                   predLabelDict_vis.get(m_i.getUniqueID()) != 1)
                    continue;

                for(int j=i+1; j<mentions.size(); j++){
                    Mention m_j = mentions.get(j);
                    if(predLabelDict_vis.containsKey(m_j.getUniqueID()) &&
                       predLabelDict_vis.get(m_j.getUniqueID()) != 1)
                        continue;

                    String id_ij = Document.getMentionPairStr(m_i, m_j);
                    String id_ji = Document.getMentionPairStr(m_j, m_i);
                    if(predLabelDict_rel.containsKey(id_ij) && predLabelDict_rel.get(id_ij) == 1 ||
                       predLabelDict_rel.containsKey(id_ji) && predLabelDict_rel.get(id_ji) == 1) {
                        String chainID_i = mentionChainIdDict.get(m_i);
                        String chainID_j = mentionChainIdDict.get(m_j);

                        //a) if one of the mentions has an ID already and the other doesn't, copy the ID
                        if(chainID_i != null && chainID_j == null){
                            mentionChainIdDict.put(m_j, chainID_i);
                        } else if (chainID_i == null && chainID_j != null) {
                            mentionChainIdDict.put(m_i, chainID_j);
                        } //b) if neither m1 nor m2 have a chain ID, put them both in a new chain
                        else if(chainID_i == null){
                            mentionChainIdDict.put(m_i, String.valueOf(chainIdx));
                            mentionChainIdDict.put(m_j, String.valueOf(chainIdx));
                            chainIdx++;
                        } //c) if both m1 and m2 have ID's and they aren't the same, merge
                        else {
                            Set<Mention> reassigMentionSet = new HashSet<>();
                            for(Mention m : mentionChainIdDict.keySet())
                                if(mentionChainIdDict.get(m).equals(chainID_j))
                                    reassigMentionSet.add(m);
                            reassigMentionSet.forEach(m -> mentionChainIdDict.put(m, chainID_i));
                        }
                    }
                }
            }


            //Add all unassigned visual mentions as singleton chains
            mentions.removeAll(mentionChainIdDict.keySet());
            for(Mention m : mentions) {
                if(!predLabelDict_vis.containsKey(m.getUniqueID()) ||
                    predLabelDict_vis.get(m.getUniqueID()) == 1)
                    mentionChainIdDict.put(m, String.valueOf(chainIdx++));
            }

            //Invert the mention / chainID dict and store the chains
            Map<String, Set<Mention>> chainMentionDict = Util.invertMap(mentionChainIdDict);
            Set<Chain> chainSet = new HashSet<>();
            for(String chainID : chainMentionDict.keySet()){
                Chain c = new Chain(d.getID(), chainID);
                for(Mention m : chainMentionDict.get(chainID))
                    c.addMention(m);
                chainSet.add(c);
            }

            //In order to make the display look correct (with nonvisuals)
            //we want to add all predicted nonvisual mentions as chain 0
            Chain nonvisChain = new Chain(d.getID(), "0");
            for(Mention m : d.getMentionList())
                if(predLabelDict_vis.containsKey(m.getUniqueID()) &&
                   predLabelDict_vis.get(m.getUniqueID()) == 0)
                    nonvisChain.addMention(m);
            chainSet.add(nonvisChain);

            if(!chainSet.isEmpty())
                predChains.put(d.getID(), chainSet);
        }

        return predChains;
    }

    /**Returns a mapping of document IDs to
     * sets of predicted chains
     *
     * @return
     */
    public Map<String, Set<Chain>> getPredictedChains(){return _predChains;}

    /**Returns of mapping of document IDs to
     * sets of chain pairs in (sub,sup) order
     *
     * @return
     */
    public Map<String, Set<Chain[]>> getPredictedSubsetChains()
    {
        Map<String, Set<Chain[]>> docSubsetChainDict = new HashMap<>();
        for(String docID : _predChains.keySet()){
            Document d = _docDict.get(docID);
            Set<Chain[]> subsetChains = new HashSet<>();
            for(String pairID : _relationGraphs.get(docID).keySet()) {
                Mention[] pair = d.getMentionPairFromStr(pairID);

                int label = _relationGraphs.get(docID).get(pairID);
                Mention subM = null, supM = null;
                if(label == 2){
                    subM = pair[0]; supM = pair[1];
                } else if(label == 3){
                    subM = pair[1]; supM = pair[0];
                }

                if(subM != null && supM != null){
                    Chain sub = null, sup = null;

                    for(Chain c : _predChains.get(docID)){
                        if(c.getMentionSet().contains(subM))
                            sub = c;
                        if(c.getMentionSet().contains(supM))
                            sup = c;
                    }

                    if(sub != null && sup != null){
                        Chain[] chainPair = {sub, sup};
                        if(!Util.containsArr(subsetChains, chainPair))
                            subsetChains.add(chainPair);
                    }
                }
            }
            if(!subsetChains.isEmpty())
                docSubsetChainDict.put(docID, subsetChains);
        }
        return docSubsetChainDict;
    }

    /* Inference */

    /**Returns the next ILPSolver thread to start, according to
     * internal inference type,
     *
     * @return
     */
    private ILPSolverThread _getNextThread(String docID,
           Map<String, Integer> fixedLinks, int numSolverThreads)
    {
        //Set up the basic thread
        ILPSolverThread thread;
        if(_infType == RELATION || _infType == VISUAL_RELATION){
            thread = new ILPSolverThread(_docDict.get(docID).getMentionList(), numSolverThreads);
        } else {
            thread = new ILPSolverThread(_docDict.get(docID).getMentionList(), _boxDict.get(docID),
                    _infType, numSolverThreads);
        }

        //Add the fixed links, if there are any
        if(fixedLinks != null && !fixedLinks.isEmpty())
            thread.setFixedRelationLinks(fixedLinks);

        if(!_nonvisScores.isEmpty())
            thread.setNonvisScores(_nonvisScores);

        if(InferenceType.isRelationType(_infType)){
            thread.setRelationScores(_relationScores);
            if(_excludeSubset)
                thread.excludeSubset();
            if(_includeTypeConstr)
                thread.includeTypeConstraint();
        }
        if(InferenceType.isGroundingType(_infType)){
            thread.setAffinityScores(_affinityScores);
            thread.setCardinalityScores(_cardinalityScores);
            if(_excludeBoxExigence)
                thread.excludeBoxExigence();
            if(_includeTypeConstr)
                thread.includeTypeConstraint();
            if(_onlyKeepPositiveLinks)
                thread.onlyKeepPositiveLinks();
        }

        return thread;
    }

    /**Performs inference according to the internal type,
     * using numThreads processes and constraining types,
     * if applicable
     *
     * @param numThreads
     */
    public void infer(int numThreads)
    {
        /* If this is a pairwise or combined inference, perform
         *  pronominal coreference resolution
         */
        Map<String, Map<String, Integer>> fixedCorefLinks = new HashMap<>();

        /*Perform inference, according to our type*/
        Logger.log("Solving the ILP for " + String.valueOf(_infType) + " inference");
        Set<String> documentIDs = _docDict.keySet();
        if((double)_relationGraphs.keySet().size()/(double)documentIDs.size() > 0.9 ||
           (double)_groundingGraphs.keySet().size()/(double)documentIDs.size() > 0.9){
            //If this is a run where most (>90%) of our documents have already
            //been solved, we can assume these are the rare, complex cases on
            //which we failed; try them one at a time, using all available threads
            List<String> docIds = new ArrayList<>(documentIDs);
            docIds.removeAll(_relationGraphs.keySet());
            docIds.removeAll(_groundingGraphs.keySet());
            _infer(docIds, fixedCorefLinks, 1, numThreads);
        } else if(!InferenceType.isJointType(_infType)){
            //If this is not three-way joint inference, we can run one document
            //per thread, giving gurobi one thread, and itll be done relatively
            //quickly
            List<String> docIds = new ArrayList<>(documentIDs);
            docIds.removeAll(_relationGraphs.keySet());
            docIds.removeAll(_groundingGraphs.keySet());
            _infer(docIds, fixedCorefLinks, numThreads, 1);
        } else {
            //If this is three-way inference, however, we assume that we have a
            //number of threads that's divisible by four (I've been running these on
            //24 threads) and we're going to split the images into batches, based
            //on the size of the graph |M|^2 + |M||B|;
            //this roughly corresponds to 0.3, 0.3, 0.2, 0.1

            //Get an in-order listing of docIDs, for each complexity bracket
            String[] complexities = {"simple", "moderate", "complex", "intractable"};
            DoubleDict<String> complexityDict = new DoubleDict<>();
            for(Document d : _docDict.values())
                complexityDict.increment(d.getID(),
                        d.getMentionList().size() * d.getMentionList().size() +
                                d.getMentionList().size() * d.getBoundingBoxSet().size());
            Map<String, List<String>> docIdDict = new HashMap<>();
            for(String complexityStr : complexities)
                docIdDict.put(complexityStr, new ArrayList<>());
            for(String docID : complexityDict.keySet()){
                int complexity = (int)complexityDict.get(docID);
                String complexityStr;
                if(complexity < 400)
                    complexityStr = complexities[0];
                else if(complexity < 700)
                    complexityStr = complexities[1];
                else if(complexity < 1200)
                    complexityStr = complexities[2];
                else
                    complexityStr = complexities[3];
                docIdDict.get(complexityStr).add(docID);
            }

            //vary the number of threads to run ourselves / to give gurobi
            //based on the complexity bracket
            for(int i=0; i<complexities.length; i++){
                int numThreads_doc = 0, numThreads_solver = 0;
                switch(i){
                    case 0: numThreads_doc = numThreads;
                            numThreads_solver = 1;
                        break;
                    case 1: numThreads_doc = numThreads / 4;
                            numThreads_solver = 4;
                        break;
                    case 2: numThreads_doc = 4;
                            numThreads_solver = numThreads / 4;
                        break;
                    case 3: numThreads_doc = 1;
                            numThreads_solver = numThreads;
                        break;
                }

                String complexityStr = complexities[i];
                List<String> docIds = new ArrayList<>(docIdDict.get(complexityStr));

                //remove all previously visited doc IDs from this list
                docIds.removeAll(_relationGraphs.keySet());
                docIds.removeAll(_groundingGraphs.keySet());
                docIds.removeAll(_visualGraphs.keySet());

                Logger.log("Running inference for %s documents "+
                           "(%d docs; %d solvers; %d thread(s) each)",
                        complexityStr.toUpperCase(), docIds.size(),
                        numThreads_doc, numThreads_solver);

                _infer(docIds, fixedCorefLinks, numThreads_doc, numThreads_solver);
            }
        }
        Logger.log("Inference complete");

        //Finally, if this has been relation or combined inference,
        //convert our graphs to predicted chains
        if(InferenceType.isRelationType(_infType)){
            Map<String, Integer> predLabels_rel = new HashMap<>();
            Map<String, Integer> predLabels_vis = new HashMap<>();
            for(Map<String, Integer> labelDict : _relationGraphs.values())
                labelDict.forEach((k,v) -> predLabels_rel.put(k, v));
            if(InferenceType.isVisualType(_infType))
                for(Map<String, Integer> labelDict : _visualGraphs.values())
                    labelDict.forEach((k,v) -> predLabels_vis.put(k, v));

            _predChains = _buildChainsFromPredLabels(predLabels_rel, predLabels_vis);
        }

        Logger.log("Failed to find solutions for %d images", _failedImgs.size());
        for(String failedImg : _failedImgs)
            System.out.println(failedImg);
        Logger.log("Fell back to individual inference for %d images", _fallbackImgs.size());
        for(String fallbackImg : _fallbackImgs)
            System.out.println(fallbackImg);
    }

    private void _infer(List<String> docIds, Map<String, Map<String, Integer>> fixedLinks,
                        int numThreads_docs, int numThreads_solver)
    {
        int docIdx = 0, threadIdx = 0;
        Thread[] threadPool = new Thread[numThreads_docs];
        while(threadIdx < numThreads_docs && docIdx < docIds.size()){
            String docID = docIds.get(docIdx);
            threadPool[threadIdx] = _getNextThread(docID,
                    fixedLinks.get(docID), numThreads_solver);
            threadPool[threadIdx].start();
            docIdx++; threadIdx++;
        }

        //keep processing threads until theyre all dead and we've gone through all the documents
        boolean foundLiveThread = true;
        while(docIdx < docIds.size() || foundLiveThread) {
            foundLiveThread = false;
            for(int i=0; i<numThreads_docs; i++) {
                //if we didn't fill the thread array, just skip this index
                if(threadPool[i] == null)
                    continue;

                if(threadPool[i].isAlive()) {
                    foundLiveThread = true;
                } else {
                    Logger.logStatus("Processed %d docs (%.2f%%)",
                            docIdx, 100.0*(double)docIdx / docIds.size());

                    //if this is a dead thread, store either its graphs
                    //or its ID, depending on whether it found a solution
                    ILPSolverThread ist = (ILPSolverThread)threadPool[i];
                    if(ist.foundSolution()){
                        _saveUpdatedGraph(ist);
                        if(ist.isFallbackSolution()){
                            _fallbackImgs.add(ist.getDocID());
                            Logger.log("WARNING: Fallback solution for " + ist.getDocID());
                        }
                    } else {
                        _failedImgs.add(ist.getDocID());
                        Logger.log("ERROR: failed to solve " + ist.getDocID());
                    }
                    threadPool[i] = null;

                    //independently, if we found a dead thread and we
                    //still have image IDs to iterate through, swap this
                    //dead one out for a live one
                    if(docIdx < docIds.size()) {
                        String docID = docIds.get(docIdx);
                        threadPool[i] = _getNextThread(docID, fixedLinks.get(docID), numThreads_solver);
                        threadPool[i].start();
                        foundLiveThread = true;
                        docIdx++;
                    }
                }
            }

            //before we check for threadlife again, let's
            //sleep for a 50 ms so we don't burn
            try{Thread.sleep(50);}
            catch(InterruptedException iEx){/*do nothing*/}
        }

        //go through the thread pool one last time to
        //collect the last of our threads
        for(int i=0; i<numThreads_docs; i++) {
            if(threadPool[i] != null){
                ILPSolverThread ist = (ILPSolverThread)threadPool[i];
                if(ist.foundSolution()){
                    _saveUpdatedGraph(ist);
                } else {
                    _failedImgs.add(ist.getDocID());
                    Logger.log("ERROR: failed to solve " + ist.getDocID());
                }
            }
        }

    }

    /**Adds the graphs present in the ILP solver thread to the
     * internal dictionary and updates the saved version of that
     * dictionary
     *
     * @param ist
     */
    private void _saveUpdatedGraph(ILPSolverThread ist)
    {
        //Add the new graphs to the dictionary and update our saved objects
        if(!_relationGraphs.containsKey(ist.getDocID())){
            _relationGraphs.put(ist.getDocID(), ist.getRelationGraph());
            FileIO.writeObject(_relationGraphs, _graphRoot + "_REL.obj");
        }
        if(!_groundingGraphs.containsKey(ist.getDocID())) {
            _groundingGraphs.put(ist.getDocID(), ist.getGroundingGraph());
            FileIO.writeObject(_groundingGraphs, _graphRoot + "_GRND.obj");
        }
        if(!_visualGraphs.containsKey(ist.getDocID())){
            _visualGraphs.put(ist.getDocID(), ist.getVisualGraph());
            FileIO.writeObject(_visualGraphs, _graphRoot + "_VIS.obj");
        }
    }

    /**InferenceType specifies the type of inference
     * we're running. At the time of this writing (12/17)
     * there are three main inference modules
     *      visual
     *      relation
     *      grounding
     * Which give us the following two-way combinations of interest
     *      visual_relation
     *      visual_grounding
     *      relation_grounding
     * The natural three-way joint inference
     *      visual_relation_grounding
     * And the following sequential modes
     *      grounding_then_relation
     *      grounding_then_visual_relation
     *      relation_then_grounding
     *      relation_then_visual_grounding
     *
     * Note that, naturally, there are other combination we don't include,
     * since visual prediction is generally unhelpful
     */
    public enum InferenceType {
        VISUAL, RELATION, GROUNDING,
        VISUAL_RELATION, VISUAL_GROUNDING, RELATION_GROUNDING,
        VISUAL_RELATION_GROUNDING,
        GROUNDING_THEN_RELATION, GROUNDING_THEN_VISUAL_RELATION,
        RELATION_THEN_GROUNDING, RELATION_THEN_VISUAL_GROUNDING;

        private static Set<InferenceType> _visualTypes, _relationTypes, _groundingTypes;
        private static Set<InferenceType> _jointTypes; // Types that require inference over
                                                       // at least two graphs
        static {
            InferenceType[] visualTypeArr = {VISUAL, VISUAL_RELATION, VISUAL_GROUNDING,
                 VISUAL_RELATION_GROUNDING, GROUNDING_THEN_VISUAL_RELATION,
                 RELATION_THEN_VISUAL_GROUNDING};
            _visualTypes = new HashSet<>(Arrays.asList(visualTypeArr));

            InferenceType[] relationTypeArr = {RELATION, VISUAL_RELATION, VISUAL_RELATION_GROUNDING,
                GROUNDING_THEN_RELATION, GROUNDING_THEN_VISUAL_RELATION, RELATION_GROUNDING,
                RELATION_THEN_GROUNDING, RELATION_THEN_VISUAL_GROUNDING};
            _relationTypes = new HashSet<>(Arrays.asList(relationTypeArr));

            InferenceType[] groundingTypeArr = {GROUNDING, VISUAL_GROUNDING, VISUAL_RELATION_GROUNDING,
                GROUNDING_THEN_RELATION, GROUNDING_THEN_VISUAL_RELATION, RELATION_GROUNDING,
                RELATION_THEN_GROUNDING, RELATION_THEN_VISUAL_GROUNDING};
            _groundingTypes = new HashSet<>(Arrays.asList(groundingTypeArr));

            InferenceType[] jointTypeArr = {VISUAL_RELATION, VISUAL_GROUNDING, RELATION_GROUNDING,
                VISUAL_RELATION_GROUNDING, GROUNDING_THEN_VISUAL_RELATION, RELATION_GROUNDING,
                RELATION_THEN_VISUAL_GROUNDING};
            _jointTypes = new HashSet<>(Arrays.asList(jointTypeArr));
        }

        public static boolean isVisualType(InferenceType t){return _visualTypes.contains(t);}
        public static boolean isRelationType(InferenceType t){return _relationTypes.contains(t);}
        public static boolean isGroundingType(InferenceType t){return _groundingTypes.contains(t);}
        public static boolean isJointType(InferenceType t){return _jointTypes.contains(t);}
    }
}
