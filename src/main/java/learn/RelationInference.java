package learn;

import statistical.ScoreDict;
import structures.Chain;
import structures.Document;
import structures.Mention;
import utilities.DoubleDict;
import utilities.Logger;
import utilities.StringUtil;
import utilities.Util;

import java.util.*;


/**The RelationInference class provides static
 * functions for computing and reporting the 
 * results of the coreference chain classifier
 * 
 * @author ccervantes
 */
public class RelationInference
{
    private static Map<String, Map<Mention, Map<Mention, Integer>>> _docGraphDict;
    private static Collection<Document> _docSet;
    private static Map<String, Set<Chain>> _predChains;

    /**Perform interfence on the given documents, relation scoreDict,
     * using numThreads at a time. Stores the graph of visual mentions
     * internally, one per document
     *
     * @param docSet
     * @param scoreDict
     * @param numThreads
     * @param scoreDict_nonvis
     */
    public static void infer(Collection<Document> docSet, Map<String, double[]> scoreDict,
                             int numThreads, BinaryClassifierScoreDict scoreDict_nonvis)
    {
        //store the doc set
        _docSet = docSet;
        Map<String, Document> docDict = new HashMap<>();
        for(Document d : docSet)
            docDict.put(d.getID(), d);

        //Associate docIDs with non-pronominal visual Mention lists
        Map<String, List<Mention>> docMentionDict = new HashMap<>();
        Set<String> nonvisualMentions = new HashSet<>();
        if(scoreDict_nonvis == null){
            for(Document d : _docSet){
                docMentionDict.put(d.getID(), new ArrayList<>());
                for(Mention m : d.getMentionList()){
                    if(m.getPronounType() == Mention.PRONOUN_TYPE.NONE ||
                       m.getPronounType() == Mention.PRONOUN_TYPE.SEMI){
                        docMentionDict.get(d.getID()).add(m);
                    }
                }
            }
        } else {
            //if we have a nonvis dict, only store visual mentions
            for(Document d : _docSet){
                docMentionDict.put(d.getID(), new ArrayList<>());
                for(Mention m : d.getMentionList()){
                    if(m.getPronounType() == Mention.PRONOUN_TYPE.NONE ||
                       m.getPronounType() == Mention.PRONOUN_TYPE.SEMI){
                        if(scoreDict_nonvis.get(m) != null){
                            if(scoreDict_nonvis.get(m) > 0)
                                nonvisualMentions.add(m.getUniqueID());
                            else
                                docMentionDict.get(d.getID()).add(m);
                        } else {
                            Logger.log("ERROR: found no score for " + m.getUniqueID() +
                                       "(" + m.toString() + ")");
                        }
                    }
                }
            }
        }
        List<String> docIdList = new ArrayList<>(docMentionDict.keySet());

        //Perform rule-based coref and fix these links during inference
        Map<String, Set<String>> pronomCoref = new HashMap<>();
        for(Document d : docSet){
            Set<String> pronomPairs = ClassifyUtil.pronominalCoref(d);
            if(!pronomPairs.isEmpty())
                pronomCoref.put(d.getID(), pronomPairs);
        }
        Map<String, Map<String, Integer>> fixedCorefLinks = new HashMap<>();
        for(String docID : pronomCoref.keySet()){
            fixedCorefLinks.put(docID, new HashMap<>());
            for(String pairStr : pronomCoref.get(docID))
                fixedCorefLinks.get(docID).put(pairStr, 1);
        }

        Logger.log("Solving ILP"); //create our first set of threads and start them
        int docIdx = 0;
        Thread[] threadPool = new Thread[numThreads];
        for(int i=0; i<numThreads; i++) {
            String docID = docIdList.get(docIdx);
            threadPool[i] = new ILPSolverThread(docMentionDict.get(docID), scoreDict, fixedCorefLinks.get(docID));
            threadPool[i].start();
            docIdx++;
        }

        //keep processing threads until theyre all dead and we've gone through all the documents
        _docGraphDict = new HashMap<>();
        boolean foundLiveThread = true;
        while(docIdx < docIdList.size() || foundLiveThread) {
            foundLiveThread = false;
            for(int i=0; i<numThreads; i++) {
                if(threadPool[i].isAlive()) {
                    foundLiveThread = true;
                } else {
                    //if this is a dead thread, store it
                    ILPSolverThread ist = (ILPSolverThread)threadPool[i];
                    _docGraphDict.put(ist.getDocID(), ist.getRelationGraph());

                    //independently, if we found a dead thread and we
                    //still have image IDs to iterate through, swap this
                    //dead one out for a live one
                    if(docIdx < docIdList.size()) {
                        Logger.logStatus("Processed %d docs (%.2f%%)",
                                docIdx, 100.0*(double)docIdx / docIdList.size());

                        String docID = docIdList.get(docIdx);
                        threadPool[i] = new ILPSolverThread(docMentionDict.get(docID), scoreDict, fixedCorefLinks.get(docID));
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
        for(int i=0; i<numThreads; i++) {
            ILPSolverThread ist = (ILPSolverThread)threadPool[i];
            if(!_docGraphDict.containsKey(ist.getDocID()))
                _docGraphDict.put(ist.getDocID(), ist.getRelationGraph());
        }

        //Finally, initialize the graph as the predicted chains
        _initPredictedChains();
    }

    /**Reads the internal graphs (one per document)
     * and stores the results as coreference chains
     */
    private static void _initPredictedChains()
    {
        //Iterate through the graph, storing mentions with chain IDs
        _predChains = new HashMap<>();
        for(String docID : _docGraphDict.keySet()){
            Map<Mention, String> mentionChainIdDict = new HashMap<>();
            int chainIdx = 1;
            Set<Mention> mentionSet = new HashSet<>();
            for(Mention m1 : _docGraphDict.get(docID).keySet()){
                mentionSet.add(m1);
                for(Mention m2 : _docGraphDict.get(docID).get(m1).keySet()){
                    mentionSet.add(m2);
                    if(_docGraphDict.get(docID).get(m1).get(m2) == 1){
                        String chainID_1 = mentionChainIdDict.get(m1);
                        String chainID_2 = mentionChainIdDict.get(m2);

                        //a) if one of the mentions has an ID already and the other doesn't, copy the ID
                        if(chainID_1 != null && chainID_2 == null){
                            mentionChainIdDict.put(m2, chainID_1);
                        } else if (chainID_1 == null && chainID_2 != null) {
                            mentionChainIdDict.put(m1, chainID_2);
                        } //b) if neither m1 nor m2 have a chain ID, put them both in a new chain
                        else if(chainID_1 == null){
                            mentionChainIdDict.put(m1, String.valueOf(chainIdx));
                            mentionChainIdDict.put(m2, String.valueOf(chainIdx));
                            chainIdx++;
                        } //c) if both m1 and m2 have ID's and they aren't the same, merge
                        else {
                            Set<Mention> reassigMentionSet = new HashSet<>();
                            for(Mention m : mentionChainIdDict.keySet())
                                if(mentionChainIdDict.get(m).equals(chainID_2))
                                    reassigMentionSet.add(m);
                            reassigMentionSet.forEach(m -> mentionChainIdDict.put(m, chainID_1));
                        }
                    }
                }
            }
            //Add all unassigned mentions as singleton chains (we assume all mentions
            //in the graph are visual)
            mentionSet.removeAll(mentionChainIdDict.keySet());
            for(Mention m : mentionSet){
                mentionChainIdDict.put(m, String.valueOf(chainIdx++));
            }

            //Invert the mention / chainID dict and store the chains
            Map<String, Set<Mention>> chainMentionDict = Util.invertMap(mentionChainIdDict);
            Set<Chain> chainSet = new HashSet<>();
            for(String chainID : chainMentionDict.keySet()){
                Chain c = new Chain(docID, chainID);
                for(Mention m : chainMentionDict.get(chainID))
                    c.addMention(m);
                chainSet.add(c);
            }
            if(!chainSet.isEmpty())
                _predChains.put(docID, chainSet);
        }
    }

    /**Returns a mapping of document IDs to sets of predicted chains
     *
     * @return
     */
    public static Map<String, Set<Chain>> getPredictedChains(){return _predChains;}

    /**Evaluates the links in the graph, considering only
     * those links between visual mentions
     *
     * @return
     */
    public static ScoreDict<Integer> evaluateGraph()
    {
        ScoreDict<Integer> linkScores = new ScoreDict<>();
        for(Document d : _docSet){
            Set<String> subsetMentions = d.getSubsetMentions();
            List<Mention> mentionList = d.getMentionList();
            for(int i=0; i<mentionList.size(); i++) {
                Mention m1 = mentionList.get(i);
                if (m1.getPronounType() != Mention.PRONOUN_TYPE.NONE && m1.getPronounType() != Mention.PRONOUN_TYPE.SEMI)
                    continue;

                if (m1.getChainID().equals("0"))
                    continue;

                for (int j = i + 1; j < mentionList.size(); j++) {
                    Mention m2 = mentionList.get(j);
                    if (m2.getPronounType() != Mention.PRONOUN_TYPE.NONE && m2.getPronounType() != Mention.PRONOUN_TYPE.SEMI)
                        continue;

                    if (m2.getChainID().equals("0"))
                        continue;

                    String id_ij = Document.getMentionPairStr(m1, m2, true, true);
                    String id_ji = Document.getMentionPairStr(m2, m1, true, true);
                    int gold_ij = 0, gold_ji = 0;
                    if(m1.getChainID().equals(m2.getChainID())){
                        gold_ij = 1; gold_ji = 1;
                    } else if(subsetMentions.contains(id_ij)){
                        gold_ij = 2; gold_ji = 3;
                    } else if(subsetMentions.contains(id_ji)){
                        gold_ij = 3; gold_ji = 2;
                    }

                    int pred_ij = 0;
                    if(_docGraphDict.get(d.getID()).containsKey(m1))
                        if(_docGraphDict.get(d.getID()).get(m1).containsKey(m2))
                            pred_ij = _docGraphDict.get(d.getID()).get(m1).get(m2);
                    int pred_ji = 0;
                    if(_docGraphDict.get(d.getID()).containsKey(m2))
                        if(_docGraphDict.get(d.getID()).get(m2).containsKey(m1))
                            pred_ji = _docGraphDict.get(d.getID()).get(m2).get(m1);

                    linkScores.increment(gold_ij, pred_ij);
                    linkScores.increment(gold_ji, pred_ji);
                }
            }
        }
        return linkScores;
    }

    public static void printPairwiseConfusion()
    {
        Map<String, DoubleDict<String>> mentionPairTable = new HashMap<>();
        mentionPairTable.put("n|n", new DoubleDict<>());
        mentionPairTable.put("c|c", new DoubleDict<>());
        mentionPairTable.put("b|p", new DoubleDict<>());
        for(Document d : _docSet){
            Set<String> subsetMentions = d.getSubsetMentions();
            List<Mention> mentionList = d.getMentionList();
            for(int i=0; i<mentionList.size(); i++) {
                Mention m1 = mentionList.get(i);
                if(m1.getChainID().equals("0"))
                    continue;

                for (int j = i + 1; j < mentionList.size(); j++) {
                    Mention m2 = mentionList.get(j);
                    if (m2.getChainID().equals("0"))
                        continue;

                    String pred_ij = "n";
                    if (_docGraphDict.get(d.getID()).containsKey(m1)){
                        if (_docGraphDict.get(d.getID()).get(m1).containsKey(m2)) {
                            switch(_docGraphDict.get(d.getID()).get(m1).get(m2)){
                                case 0: pred_ij = "n"; break;
                                case 1: pred_ij = "c"; break;
                                case 2: pred_ij = "b"; break;
                                case 3: pred_ij = "p"; break;
                            }
                        }
                    }
                    String pred_ji = "n";
                    if (_docGraphDict.get(d.getID()).containsKey(m2)){
                        if (_docGraphDict.get(d.getID()).get(m2).containsKey(m1)) {
                            switch(_docGraphDict.get(d.getID()).get(m2).get(m1)){
                                case 0: pred_ji = "n"; break;
                                case 1: pred_ji = "c"; break;
                                case 2: pred_ji = "b"; break;
                                case 3: pred_ji = "p"; break;
                            }
                        }
                    }
                    String pred = StringUtil.getAlphabetizedPair(pred_ij, pred_ji);

                    String id_ij = Document.getMentionPairStr(m1, m2, true, true);
                    String id_ji = Document.getMentionPairStr(m2, m1, true, true);
                    String gold = "n|n";
                    if(m1.getChainID().equals(m2.getChainID())){
                        gold = "c|c";
                    } else if(subsetMentions.contains(id_ij) || subsetMentions.contains(id_ji)){
                        gold = "b|p";
                    }

                    mentionPairTable.get(gold).increment(pred);
                }
            }
        }
        Set<String> predSet = new HashSet<>();
        for(String gold : mentionPairTable.keySet())
            predSet.addAll(mentionPairTable.get(gold).keySet());
        List<String> predList = new ArrayList<>(predSet);

        List<List<String>> table = new ArrayList<>();
        List<String> columns = new ArrayList<>(predList);
        columns.add(0, "");
        table.add(columns);
        for(String gold : mentionPairTable.keySet()){
            List<String> row = new ArrayList<>();
            row.add(gold);
            for(String pred : predList){
                int count = (int)mentionPairTable.get(gold).get(pred);
                double perc = 100.0 * count / mentionPairTable.get(gold).getSum();
                row.add(String.format("%d (%.1f%%)", count, perc));
            }
            table.add(row);
        }
        System.out.println(StringUtil.toTableStr(table));
    }

    /**Returns a set of predicted subset pairs, given by the graph
     *
     * @return
     */
    public static Map<String, Set<String>> getPredictedSubsetPairs()
    {
        Map<String, Set<String>> docSubsetPairs = new HashMap<>();
        for(String docID : _docGraphDict.keySet()){
            Set<String> subsetPairs = new HashSet<>();
            for(Mention m1 : _docGraphDict.get(docID).keySet()) {
                for (Mention m2 : _docGraphDict.get(docID).get(m1).keySet()) {
                    int label = _docGraphDict.get(docID).get(m1).get(m2);
                    if (label == 2)
                        subsetPairs.add(Document.getMentionPairStr(m1, m2, true, true));
                    else if (label == 3)
                        subsetPairs.add(Document.getMentionPairStr(m2, m1, true, true));
                }
            }
            if(!subsetPairs.isEmpty())
                docSubsetPairs.put(docID, subsetPairs);
        }
        return docSubsetPairs;
    }

    public static Map<String, Set<Chain[]>> getPredictedSubsetChains()
    {
        Map<String, Set<Chain[]>> docSubsetChainDict = new HashMap<>();
        for(String docID : _predChains.keySet()){
            Set<Chain[]> subsetChains = new HashSet<>();

            for(Mention m1 : _docGraphDict.get(docID).keySet()) {
                for (Mention m2 : _docGraphDict.get(docID).get(m1).keySet()) {
                    int label = _docGraphDict.get(docID).get(m1).get(m2);
                    Mention subM = null, supM = null;
                    if(label == 2){
                        subM = m1; supM = m2;
                    } else if(label == 3){
                        subM = m2; supM = m1;
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
                            Chain[] pair = {sub, sup};
                            if(!Util.containsArr(subsetChains, pair))
                                subsetChains.add(pair);
                        }
                    }
                }
            }
            if(!subsetChains.isEmpty())
                docSubsetChainDict.put(docID, subsetChains);
        }
        return docSubsetChainDict;
    }
}
