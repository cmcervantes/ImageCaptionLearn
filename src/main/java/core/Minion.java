package core;

import learn.BinaryClassifierScoreDict;
import learn.ClassifyUtil;
import learn.FeatureVector;
import nlptools.WordnetUtil;
import out.OutTable;
import structures.*;
import utilities.*;

import java.awt.geom.Area;
import java.io.*;
import java.util.*;
import java.util.function.Function;

import static core.Overlord.boxFeatureDir;
import static core.Overlord.dbPath;

public class Minion
{
    /**Exports a csv (at given outRoot) where the mentions
     * of the given document have been filtered using the
     * given filter; intended for use in situations where
     * specific mentions from a document are desired
     *
     * @param docSet
     * @param filter
     * @param outRoot
     */
    public static void export_filteredMentions(Collection<Document> docSet,
           Function<Document, Set<Mention>> filter, String outRoot)
    {
        OutTable ot = new OutTable("doc_id", "caption_idx",
                "mention_idx", "lex_type", "chain_id",
                "mention", "caption");
        for(Document d : docSet){
            for(Mention m : filter.apply(d)){
                ot.addRow(d.getID(), m.getCaptionIdx(), m.getIdx(),
                          m.getLexicalType(), m.getChainID(), m.toString(),
                          d.getCaption(m.getCaptionIdx()).toEntitiesString());
            }
        }
        ot.writeToCsv(outRoot);
    }

    /**Exports a csv (at given outRoot) where the mentions of
     * the given document have been filtered using the given
     * filter; intended for use in situation where
     * specific mention pairs (along with some case information)
     * are desired
     *
     * @param docSet
     * @param filter
     * @param outRoot
     */
    public static void export_filteredMentionPairCases(Collection<Document> docSet,
           Function<Document, Map<Mention[], String>> filter, String outRoot)
    {
        OutTable ot = new OutTable("doc_id", "caption_1_idx",
                "mention_1_idx", "lex_type_1", "chain_id_1",
                "caption_2_idx", "mention_2_idx", "lex_type_2",
                "chain_id_2", "mention_1", "mention_2", "case",
                "caption_1", "caption_2");
        for(Document d : docSet){
            Map<Mention[], String> caseDict = filter.apply(d);
            for(Mention[] pair : caseDict.keySet()){
                Mention m1 = pair[0];
                Mention m2 = pair[1];
                String caseStr = caseDict.get(pair);
                ot.addRow(d.getID(), m1.getCaptionIdx(),
                        m1.getIdx(), m1.getLexicalType(),
                        m1.getChainID(), m2.getCaptionIdx(),
                        m2.getIdx(), m2.getLexicalType(),
                        m2.getChainID(), m1.toString(),
                        m2.toString(), caseStr,
                        d.getCaption(m1.getCaptionIdx()).toEntitiesString(),
                        d.getCaption(m2.getCaptionIdx()).toEntitiesString());
            }
        }
        ot.writeToCsv(outRoot);
    }

    /**Given a document, returns a set of fully pronominal mentions for which
     * there is no intra-caption referent; intended to identify bad pronoun
     * annoations
     *
     * @param d
     * @return
     */
    public static Set<Mention> filter_nonCorefPronouns(Document d)
    {
        Set<Mention> heterogCapMentions = new HashSet<>();
        for(Caption c : d.getCaptionList()){
            //Get the number of times we see a chain in a caption
            DoubleDict<String> chainCounts = new DoubleDict<>();
            for(Mention m : c.getMentionList())
                chainCounts.increment(m.getChainID());

            //Any fully pronominal mention with a unique chain ID for its caption
            //(that is, it introduces a new referent) is suspect
            for(Mention m : c.getMentionList()){
                if(m.getPronounType() != Mention.PRONOUN_TYPE.NONE &&
                   m.getPronounType() != Mention.PRONOUN_TYPE.SEMI &&
                   m.getPronounType() != Mention.PRONOUN_TYPE.INDEFINITE){
                    if(chainCounts.get(m.getChainID()) == 1)
                        heterogCapMentions.add(m);
                }
            }
        }
        return heterogCapMentions;
    }

    /**Given a document, returns a mapping of ordered mention pairs
     * and their associated box relationships; intended to identify
     * missing gold subset pairs
     *
     * @param d
     * @return
     */
    public static Map<Mention[], String> filter_boxSubsetCases(Document d)
    {
        Map<Mention[], String> pairCases = new HashMap<>();
        List<Mention> mentions = d.getMentionList();
        for(int i=0; i<mentions.size(); i++){
            Mention m_i = mentions.get(i);
            Set<BoundingBox> boxes_i = d.getBoxSetForMention(m_i);
            for(int j=i+1; j<mentions.size(); j++) {
                Mention m_j = mentions.get(j);
                Set<BoundingBox> boxes_j = d.getBoxSetForMention(m_j);

                if(Mention.getLexicalTypeMatch(m_i, m_j) == 0)
                    continue;

                if(m_i.getChainID().equals(m_j.getChainID()))
                    continue;

                if(m_i.getChainID().equals("0") || m_j.getChainID().equals("0"))
                    continue;

                if(!boxes_i.isEmpty() && !boxes_j.isEmpty() &&
                        boxes_i.size() != boxes_j.size()){
                    Set<BoundingBox> boxes_1, boxes_2;
                    boolean ij_ordering = boxes_i.size() < boxes_j.size();
                    if(ij_ordering){
                        boxes_1 = boxes_i; boxes_2 = boxes_j;
                    } else {
                        boxes_1 = boxes_j; boxes_2 = boxes_i;
                    }

                    String caseStr = "";
                    Set<BoundingBox> intersect = new HashSet<>(boxes_1);
                    intersect.retainAll(boxes_2);
                    if(intersect.size() == boxes_1.size() &&
                            !intersect.isEmpty() &&
                            boxes_2.size() > intersect.size()) {
                        caseStr = "subset";
                    } else {
                        boolean coveredallBoxes = true;
                        for(BoundingBox b_1 : boxes_1) {
                            if(!boxes_2.contains(b_1)){
                                boolean foundMatch = false;
                                for(BoundingBox b_2 : boxes_2)
                                    if(BoundingBox.IOU(b_1, b_2) > 0.9)
                                        foundMatch = true;
                                if(!foundMatch)
                                    coveredallBoxes = false;
                            }
                        }
                        if(coveredallBoxes)
                            caseStr = "iou_subset";
                        else
                            caseStr = "unk_subset";

                        if(m_i.getLexicalType().contains("people") && m_j.getLexicalType().contains("people"))
                            caseStr += "_ppl";

                        if(m_i.getLexicalType().contains("other") || m_j.getLexicalType().contains("other"))
                            caseStr += "_other";
                    }

                    if(ij_ordering)
                        pairCases.put(new Mention[]{m_i, m_j}, caseStr);
                    else
                        pairCases.put(new Mention[]{m_j, m_i}, caseStr);
                }
            }
        }
        return pairCases;
    }

    /**Given a document, returns a mapping of unordered mention pairs
     * and their associated gold/pred pronominal coreference labelings;
     * intended for use both in coreference evaluation
     *
     * @param d
     * @return
     */
    public static Map<Mention[], String> filter_pronomCorefEval(Document d)
    {
        Map<Mention[], String> pronomCorefPairs = new HashMap<>();

        Set<String> corefPairs_pred = ClassifyUtil.pronominalCoref(d,
                new HashSet<>(d.getMentionList()));
        List<Mention> mentions = d.getMentionList();
        for(int i=0; i<mentions.size(); i++){
            Mention m_i = mentions.get(i);
            Mention.PRONOUN_TYPE pType_i = m_i.getPronounType();
            for(int j=i+1; j<mentions.size(); j++){
                Mention m_j = mentions.get(j);
                Mention.PRONOUN_TYPE pType_j = m_j.getPronounType();

                if(m_i.getCaptionIdx() != m_j.getCaptionIdx())
                    continue;

                String id_ij = Document.getMentionPairStr(m_i, m_j);
                String id_ji = Document.getMentionPairStr(m_j, m_i);

                if(pType_i != Mention.PRONOUN_TYPE.NONE && pType_i != Mention.PRONOUN_TYPE.SEMI ||
                   pType_j != Mention.PRONOUN_TYPE.NONE && pType_j != Mention.PRONOUN_TYPE.SEMI){

                    String caseStr = "";
                    if(m_i.getChainID().equals(m_j.getChainID()))
                        caseStr = "gold";
                    if(corefPairs_pred.contains(id_ij) || corefPairs_pred.contains(id_ji)){
                        if(!caseStr.isEmpty())
                            caseStr += "_";
                        caseStr += "pred";
                    }
                    if(caseStr.isEmpty())
                        caseStr = "none";

                    if(!Util.containsArr(pronomCorefPairs.keySet(), new Mention[]{m_i, m_j}) &&
                       !Util.containsArr(pronomCorefPairs.keySet(), new Mention[]{m_j, m_i})){
                        pronomCorefPairs.put(new Mention[]{m_i, m_j}, caseStr);
                    }
                }
            }
        }
        return pronomCorefPairs;
    }

    /**Exports a file at outRoot that contains hypernym branches for
     * each head word for each mention in the docSet; rootConcepts
     * contains terminal nodes (branches contain only enough
     * nodes to include a root concept if found, all concepts if not)
     *
     * @param docSet
     * @param outRoot
     */
    public static void export_hypernymBranches(Collection<Document> docSet, String outRoot,
                                               Set<String> rootConcepts)
    {
        WordnetUtil wnUtil = new WordnetUtil(Overlord.wordnetDir);
        Map<String, List<String>> lemmaBranchDict = new HashMap<>();
        DoubleDict<String> lemmaFreq = new DoubleDict<>();
        DoubleDict<String> hypFreq = new DoubleDict<>();
        for(Document d : docSet){
            for(Mention m : d.getMentionList()){
                String lemma = m.getHead().getLemma().toLowerCase();
                lemmaFreq.increment(lemma);
                Set<String> hypRoots = new HashSet<>();
                if(!lemmaBranchDict.containsKey(lemma)){
                    HypTree hypTree = wnUtil.getHypernymTree(lemma);
                    List<String> branches = new ArrayList<>();
                    for(List<HypTree.HypNode> branch : hypTree.getRootBranches(true)){
                        List<String> nodes = new ArrayList<>();
                        for(HypTree.HypNode h : branch){
                            String hStr = h.toString();
                            nodes.add(hStr);
                            if(rootConcepts.contains(hStr))
                                break;
                        }
                        String root = nodes.get(nodes.size()-1);
                        if(!rootConcepts.contains(root))
                            root = "UNK";
                        hypRoots.add(root);
                        branches.add(StringUtil.listToString(nodes, "|"));
                    }
                    lemmaBranchDict.put(lemma, branches);
                }
                for(String h : hypRoots)
                    hypFreq.increment(h, 1.0/hypRoots.size());
            }
        }
        List<String> ll = new ArrayList<>();
        for(String lemma : lemmaBranchDict.keySet()){
            StringBuilder sb = new StringBuilder();
            sb.append("\""); sb.append(lemma.replace("\"", "'")); sb.append("\"");
            sb.append(",");
            sb.append(lemmaFreq.get(lemma));
            sb.append(",");
            for(int i=0; i<lemmaBranchDict.get(lemma).size(); i++){
                sb.append("\""); sb.append(lemmaBranchDict.get(lemma).get(i)); sb.append("\"");
                if(i < lemmaBranchDict.size() - 1)
                    sb.append(",");
            }
            ll.add(sb.toString());
        }
        FileIO.writeFile(ll, outRoot, "csv", true);

        double total = hypFreq.getSum() / 100.0;
        for(String h : hypFreq.keySet())
            hypFreq.divide(h, total);
        System.out.println(hypFreq);
    }

    public static void test_cardinality()
    {
        Logger.log("Loading documents");
        DBConnector conn = new DBConnector(dbPath);
        Collection<Document> docSet = DocumentLoader.getDocumentSet(conn, 0);

        Map<String, String> examples_gold = new HashMap<>();
        examples_gold.put("two little dogs", "1|2");
        examples_gold.put("the early winter days", "null");
        examples_gold.put("blue jeans", "1|1");
        examples_gold.put("gray sweatpants", "1|1");
        examples_gold.put("red and white swimming trunks", "1|1");
        examples_gold.put("black sunglasses", "1|1");
        examples_gold.put("pink goggles", "1|1");
        examples_gold.put("a line", "1|1");
        examples_gold.put("khaki bottoms", "1|1");
        examples_gold.put("2 brunettes", "1|2");
        examples_gold.put("a three legged black and white dog", "1|1");
        examples_gold.put("three children", "1|3");
        examples_gold.put("the three men", "1|3");
        examples_gold.put("a three-man band", "1|3");
        examples_gold.put("a three wheel scooter", "1|1");
        examples_gold.put("the waters", "m");
        examples_gold.put("cinder blocks", "1|1+");
        examples_gold.put("the four children", "1|4");
        examples_gold.put("dark blue outfits", "1|1+");
        examples_gold.put("two others", "1|2");
        examples_gold.put("a red , four door car", "1|1");
        examples_gold.put("two boys soccer teams", "2|1+");
        examples_gold.put("the contents", "1|1+");
        examples_gold.put("other couples", "1+|2");
        examples_gold.put("a four-wheeler", "1|1");
        examples_gold.put("burning piles", "1|1+");
        examples_gold.put("canned drinks", "1|1+");
        examples_gold.put("dark clouds", "1|1+");
        examples_gold.put("two brown ones", "1|2");
        examples_gold.put("two people", "1|2");
        examples_gold.put("5 gallon bottles", "1|1+");
        examples_gold.put("a least four instrumentalists", "1|4+");
        examples_gold.put("the ' 9 '", "1|1");
        examples_gold.put("half past 11", "1|1");
        examples_gold.put("numbers 11", "1|1");
        examples_gold.put("the 30 yard line", "1|1");
        examples_gold.put("a 30 somethings man", "1|1");
        examples_gold.put("player 44", "1|1");
        examples_gold.put("53 seconds", "null");
        examples_gold.put("a 90-year-old woman", "1|1");
        examples_gold.put("the 2012 london olympics", "1|1");
        examples_gold.put("a nail polish bottle 745", "1|1");
        examples_gold.put("a lot of women", "1|1+");
        examples_gold.put("dozens of motor scooters", "1+|12");
        examples_gold.put("a basket of toys", "1|1+");
        examples_gold.put("a narrow set of wooden stairs", "1|1");
        examples_gold.put("a couple of old men", "1|2");
        examples_gold.put("group of people", "1|1+");
        examples_gold.put("a pair of sunglasses", "1|1");
        examples_gold.put("a band of men", "1|1+");
        examples_gold.put("a group of individuals", "1|1+");
        examples_gold.put("a number of holiday gifts", "1|1+");
        examples_gold.put("an open field of hay", "1|1");
        examples_gold.put("a large row of yellow bananas", "1|1+");
        examples_gold.put("ribbons of many colors", "1|1+");
        examples_gold.put("a pair of scissors", "1|1");
        examples_gold.put("a bale of hay", "1|1");
        examples_gold.put("starbucks coffee", "m");
        examples_gold.put("a hill of snow", "1|1");
        examples_gold.put("a type of instrument", "1|1");
        examples_gold.put("his cup of coffee", "1|1");
        examples_gold.put("a pile of potatoes", "1|1+");
        examples_gold.put("a couple ladies", "1|2");
        examples_gold.put("several men", "1|2+");
        examples_gold.put("some stores", "1|1+");
        examples_gold.put("three male and female pairs of dancers", "3|2");
        examples_gold.put("a large glass of beer", "1|1");
        examples_gold.put("a pile of dirt", "1|1");
        examples_gold.put("a set of high bleachers or stairs", "1|1");
        examples_gold.put("a type of go-cart", "1|1");
        examples_gold.put("a field of snow", "1|1");
        examples_gold.put("a group of seven asian girls", "1|7");
        examples_gold.put("a pair of pants", "1|1");
        examples_gold.put("a mug of liquid", "1|1");
        examples_gold.put("dozens of sheep", "1+|12");
        examples_gold.put("two pairs of scales", "2|2");
        examples_gold.put("a couple of soccer players", "1|2");
        examples_gold.put("a bottle of water", "1|1");
        examples_gold.put("a small body of water", "1|1");


        Logger.log("Initializing cardinality lists");
        Cardinality.initCardLists(Overlord.resourcesDir + "/collectiveNouns.txt");

        Logger.log("Re-computing and evaluating test cases");
        Map<String, String> examples_incorrect = new HashMap<>();
        Set<String> ex = new HashSet<>();
        OutTable ot = new OutTable("mention", "card", "gold_card");
        for(Document d : docSet){
            for(Mention m : d.getMentionList()){
                String normText = m.toString().toLowerCase();
                if(examples_gold.containsKey(normText) && !examples_incorrect.containsKey(normText)){
                    Cardinality c = Cardinality.parseCardinality(m.getTokenList());
                    String goldCard = examples_gold.get(normText);

                    if(!ex.contains(m.toString().toLowerCase())) {
                        ot.addRow(m.toString(), c.toString(), goldCard);
                        ex.add(m.toString().toLowerCase());
                    }


                    if(!c.toString().equals(goldCard)){
                        examples_incorrect.put(normText, c.toString());
                        System.out.print("{");
                        for(int i=0; i<m.getTokenList().size(); i++){
                            Token t = m.getTokenList().get(i);
                            System.out.print("{\"" + t.toString() + "\",\"");
                            System.out.print(t.getLemma() + "\",\"" + t.getPosTag() + "\"}");
                            if(i < m.getTokenList().size() - 1)
                                System.out.print(",");
                        }
                        System.out.println("}");
                    }
                }
            }
        }
        ot.writeToCsv("ex_card_test", true);
    }

    public static void export_subsetHeterogType(Collection<Document> docSet)
    {
        DoubleDict<String> linkDict = new DoubleDict<>();

        DoubleDict<String> heterogHeads = new DoubleDict<>();
        OutTable ot = new OutTable("cap_id", "m1", "m1_idx", "m1_head", "m1_type",
                "m2", "m2_idx", "m2_head", "m2_type", "case", "cap1", "cap2");
        for(Document d : docSet){
            List<Mention> mentionList = d.getMentionList();
            for(int i=0; i<mentionList.size(); i++){
                Mention m1 = mentionList.get(i);
                if(m1.getPronounType() != Mention.PRONOUN_TYPE.NONE)
                    continue;

                for(int j=i+1; j<mentionList.size(); j++){
                    Mention m2 = mentionList.get(j);
                    if(m2.getPronounType() != Mention.PRONOUN_TYPE.NONE)
                        continue;

                    boolean isCoref = false;
                    boolean isSubset_ab = false;
                    boolean isSubset_ba = false;
                    if(!m1.getChainID().equals("0") && !m2.getChainID().equals("0")){
                        isCoref = m1.getChainID().equals(m2.getChainID());
                        isSubset_ab = d.getBoxesAreSubset(m1, m2);
                        isSubset_ba = d.getBoxesAreSubset(m2, m1);
                    }

                    boolean typeMatch = Mention.getLexicalTypeMatch(m1,m2) > 0;
                    boolean heterogType = !typeMatch;
                    if(!typeMatch &&
                            !m1.getLexicalType().contains("other") &&
                            !m2.getLexicalType().contains("other"))
                        heterogType = false;

                    if(isCoref && heterogType){
                        linkDict.increment("coref_heterog", 2);
                    } else if(isCoref && typeMatch){
                        linkDict.increment("coref_homog", 2);
                    }

                    if(isSubset_ab && heterogType){
                        linkDict.increment("subset_ab_heterog");
                    } else if(isSubset_ab && typeMatch){
                        linkDict.increment("subset_ab_homog");
                    }

                    if(isSubset_ba && heterogType){
                        linkDict.increment("subset_ba_heterog");
                    } else if(isSubset_ba && typeMatch){
                        linkDict.increment("subset_ba_homog");
                    }

                    heterogHeads.increment(m1.getHead().toString().toLowerCase() + "-" + m1.getLexicalType());
                    heterogHeads.increment(m2.getHead().toString().toLowerCase() + "-" + m2.getLexicalType());

                    //store all heterog mention pairs that are either subset or coref
                    if(heterogType && (isCoref || isSubset_ab || isSubset_ba)){
                        String caseStr = "";
                        if(isCoref)
                            caseStr = "coref";
                        else if(isSubset_ab)
                            caseStr = "subset_ab";
                        else if(isSubset_ba)
                            caseStr = "subset_ba";
                        ot.addRow(d.getID() + "#" + m1.getCaptionIdx(), m1.toString().replace(",", ""),
                                m1.getIdx(), m1.getHead().toString(), m1.getLexicalType(), m2.toString().replace(",", ""),
                                m2.getIdx(), m2.getHead().toString(), m2.getLexicalType(), caseStr,
                                d.getCaption(m1.getCaptionIdx()).toString().replace(",", ""),
                                d.getCaption(m2.getCaptionIdx()).toString().replace(",", ""));
                    }
                }
            }
        }
        System.out.println(linkDict.toString());
        ot.writeToCsv("ex_heterog_type_pairs_dev", false);


        OutTable ot_heads = new OutTable("head", "type", "freq");
        for(String headType : heterogHeads.keySet()){
            String[] pair = headType.split("-");
            ot_heads.addRow(pair[0], pair[1], heterogHeads.get(headType));
        }
        ot_heads.writeToCsv("ex_heads_dev", false);
    }

    public static void export_penultFilter(Collection<Document> docSet)
    {
        OutTable ot = new OutTable("Img", "m_1", "m_2", "cap_1", "cap_2");
        Set<String> penult_docs = new HashSet<>();
        Set<String> penult_caps = new HashSet<>();
        Set<String> penult_mentions = new HashSet<>();
        int totalMentions = 0;
        for(Document d : docSet){
            List<Mention> mentionList = d.getMentionList();
            int numMentions = mentionList.size();
            totalMentions += numMentions;
            for(int i=0; i<numMentions; i++){
                Mention m1 = mentionList.get(i);
                int numTokens_1 = m1.getTokenList().size();
                for(int j=i+1; j<numMentions; j++){
                    Mention m2 = mentionList.get(j);
                    int numTokens_2 = m2.getTokenList().size();

                    String penult_1 = null, penult_2 = null;
                    if(numTokens_1 > 1)
                        penult_1 = m1.getTokenList().get(numTokens_1-2).toString().toLowerCase();
                    if(numTokens_2 > 1)
                        penult_2 = m2.getTokenList().get(numTokens_2-2).toString().toLowerCase();
                    String ult_1 = m1.getTokenList().get(numTokens_1-1).toString().toLowerCase();
                    String ult_2 = m2.getTokenList().get(numTokens_2-1).toString().toLowerCase();

                    if(penult_1 != null && penult_1.equals(ult_2) ||
                       penult_2 != null && penult_2.equals(ult_1)){
                        Caption c1 = d.getCaption(m1.getCaptionIdx());
                        Caption c2 = d.getCaption(m2.getCaptionIdx());
                        ot.addRow(d.getID(), m1.toString(), m2.toString(), c1.toString(), c2.toString());


                        penult_docs.add(d.getID());
                        penult_caps.add(c1.getUniqueID());
                        penult_caps.add(c2.getUniqueID());
                        penult_mentions.add(m1.getUniqueID());
                        penult_mentions.add(m2.getUniqueID());
                    }
                }
            }
        }
        ot.writeToCsv("ex_penultFilter", true);
        Logger.log("Penultimate Filter Statistics");
        System.out.printf("Documents: %d (%.2f%%)\nCaptions: %d (%.2f%%)\nMentions: %d (%.2f%%)\n",
                penult_docs.size(), 100.0 * penult_docs.size() / docSet.size(),
                penult_caps.size(), 100.0 * penult_caps.size() / (docSet.size() * 5),
                penult_mentions.size(), 100.0 * penult_mentions.size() / totalMentions);
    }

    public static void export_modSubsetFeats(Collection<Document> docSet, String dataSplit)
    {
        Logger.log("Loading vectors");
        String filename = "";
        if(dataSplit != null && dataSplit.equals("dev"))
            filename = "pairwise_dev.feats";
        else if(dataSplit != null && dataSplit.equals("train"))
            filename = "pairwise_train.feats";
        List<String> fvStrList = FileIO.readFile_lineList(filename);

        Map<String, FeatureVector> fvDict = new HashMap<>();
        for(String fvStr : fvStrList){
            FeatureVector fv = FeatureVector.parseFeatureVector(fvStr);
            fvDict.put(fv.comments, fv);
        }

        Logger.log("Computing new subset pairs");
        Set<Mention[]> subsetPairs = new HashSet<>();
        int doc_idx = 0;
        Map<String, Document> docDict = new HashMap<>();
        docSet.forEach(d -> docDict.put(d.getID(), d));
        for(Document d : docSet){
            List<Mention> mentionList = d.getMentionList();
            for(int i=0; i<mentionList.size(); i++){
                Mention m1 = mentionList.get(i);
                if(m1.getPronounType() != Mention.PRONOUN_TYPE.NONE)
                    continue;

                for(int j=i+1; j<mentionList.size(); j++){
                    Mention m2 = mentionList.get(j);
                    if(m2.getPronounType() != Mention.PRONOUN_TYPE.NONE)
                        continue;

                    if(m1.getChainID().equals(m2.getChainID()))
                        continue;
                    if(m1.getChainID().equals("0") && m2.getChainID().equals("0"))
                        continue;

                    Set<BoundingBox> boxes_1 = d.getBoxSetForMention(m1);
                    Set<BoundingBox> boxes_2 = d.getBoxSetForMention(m2);
                    if(boxes_1.isEmpty() || boxes_2.isEmpty())
                        continue;

                    Set<String> collectives = new HashSet<>(FileIO.readFile_lineList(Overlord.resourcesDir + "collectiveNouns.txt"));
                    boolean m1_coll = false;
                    String m1_norm = m1.toString().toLowerCase().trim();
                    for(String coll : collectives){
                        if(m1_norm.endsWith(coll) || m1_norm.contains(coll + " of ")){
                            m1_coll = true;
                            break;
                        }
                    }
                    boolean m2_coll = false;
                    String m2_norm = m2.toString().toLowerCase().trim();
                    for(String coll : collectives){
                        if(m2_norm.endsWith(coll) || m2_norm.contains(coll + " of ")){
                            m2_coll = true;
                            break;
                        }
                    }
                    if(Mention.getLexicalTypeMatch(m1, m2) > 0){
                        boolean m1_subj_m2_obj = false, m2_subj_m1_obj = false;
                        Caption c1 = d.getCaption(m1.getCaptionIdx());
                        Caption c2 = d.getCaption(m2.getCaptionIdx());

                        //We want to drop any candidates for which
                        //m1 is a subject of a verb that m2 is an object of
                        //(and vice versa) taking coreference information
                        //into account (such that if m1_verb_mX and m2
                        //is coref with mX, the relationship holds)
                        if(c1.equals(c2)){
                            Chunk subj1 = c1.getSubjectOf(m1);
                            Chunk subj2 = c1.getSubjectOf(m2);
                            if(subj1 != null && subj1.equals(c1.getObjectOf(m2)))
                                m1_subj_m2_obj = true;
                            else if(subj2 != null && subj2.equals(c1.getObjectOf(m1)))
                                m2_subj_m1_obj = true;
                        } else {
                            Chain ch1 = null, ch2 = null;
                            for(Chain ch : d.getChainSet()){
                                if(ch.getMentionSet().contains(m1))
                                    ch1 = ch;
                                else if(ch.getMentionSet().contains(m2))
                                    ch2 = ch;
                            }

                            if(ch1 != null && ch2 != null){
                                Chunk subj1 = c1.getSubjectOf(m1);
                                Chunk subj2 = c1.getSubjectOf(m2);
                                if(subj1 != null){
                                    //For each mention (m') to which m2 corefers, if
                                    //      a) m' is from caption c1
                                    //      b) m' is an object of verb v
                                    //      c) m1 is the subject of verb v
                                    for(Mention mPrime : ch2.getMentionSet()){
                                        if(mPrime.getCaptionIdx() == c1.getIdx()){
                                            if(subj1.equals(c1.getObjectOf(mPrime)))
                                                m1_subj_m2_obj = true;
                                        }
                                    }
                                }
                                if(subj2 != null){
                                    for(Mention mPrime : ch2.getMentionSet()){
                                        if(mPrime.getCaptionIdx() == c2.getIdx())
                                            if(subj2.equals(c2.getObjectOf(mPrime)))
                                                m2_subj_m1_obj = true;
                                    }
                                }
                            }
                        }

                        //only consider subsets between m1 and m2 where
                        //  a) m1 is collective
                        //  b) m2 is _not_ collective
                        //  c) the [m1 verb m2] relationship does not hold
                        if(m1_coll && !m2_coll && !m1_subj_m2_obj){

                            //m1 is collective. Check if it's boxes contain all those in m2
                            Area a1 = new Area();
                            boxes_1.forEach(b -> a1.add(new Area(b.getRec())));
                            boolean fullCoverage = true;
                            for(BoundingBox b2 : boxes_2)
                                if(!a1.contains(b2.getRec()))
                                    fullCoverage = false;
                            if(fullCoverage)
                                subsetPairs.add(new Mention[]{m2, m1});
                        } else if(m2_coll && !m1_coll && !m2_subj_m1_obj){
                            Area a2 = new Area();
                            boxes_2.forEach(b -> a2.add(new Area(b.getRec())));
                            boolean fullCoverage = true;
                            for(BoundingBox b1 : boxes_1)
                                if(!a2.contains(b1.getRec()))
                                    fullCoverage = false;
                            if(fullCoverage)
                                subsetPairs.add(new Mention[]{m1, m2});
                        } else if(boxes_1.containsAll(boxes_2) && boxes_2.containsAll(boxes_1)){
                            if(m1.getCardinality().getValue() < m2.getCardinality().getValue()){
                                subsetPairs.add(new Mention[]{m1, m2});
                            } else if(m2.getCardinality().getValue() < m1.getCardinality().getValue()){
                                subsetPairs.add(new Mention[]{m2, m1});
                            }
                        }
                    }
                }
            }
            doc_idx++;
            Logger.logStatus("%d complete (%.2f%%)", doc_idx, 100.0 * doc_idx / docSet.size());
        }

                /*
                Logger.log("Writing prev null examples");
                OutTable ot = new OutTable("m1", "m2", "cap1", "cap2");
                for(Mention[] pair : subsetPairs){
                    Mention m1 = pair[0]; Mention m2 = pair[1];
                    String pairID = Document.getMentionPairStr(m1, m2);
                    if(fvDict.get(pairID).label == 0){
                        String cap1 = docDict.get(m1.getDocID()).getCaption(m1.getCaptionIdx()).toString();
                        String cap2 = docDict.get(m2.getDocID()).getCaption(m2.getCaptionIdx()).toString();
                        ot.addRow(m1.toString(), m2.toString(), cap1, cap2);
                    }
                }
                ot.writeToCsv("ex_new_subset_prev_null", false);*/

        DoubleDict<Integer> distro = new DoubleDict<>();
        for(FeatureVector fv : fvDict.values())
            distro.increment((int)fv.label);
        System.out.println(distro.toString());

        for(Mention[] pair : subsetPairs){
            Mention m1 = pair[0]; Mention m2 = pair[1];
            String pairID_12 = Document.getMentionPairStr(m1, m2);
            String pairID_21 = Document.getMentionPairStr(m2, m1);
            if(fvDict.get(pairID_12).label == 0){
                fvDict.get(pairID_12).label = 2;
                fvDict.get(pairID_21).label = 3;
            }
        }

        distro = new DoubleDict<>();
        for(FeatureVector fv : fvDict.values())
            distro.increment((int)fv.label);
        System.out.println(distro.toString());

        List<String> ll_fv = new ArrayList<>();
        fvDict.values().forEach(fv -> ll_fv.add(fv.toString()));
        FileIO.writeFile(ll_fv, filename.replace(".feats", "_mod"), "feats", false);
    }

    public static void print_pairwiseConfusion(Collection<Document> docSet,
                                               String scoresFile)
    {
        List<String> ll_pairwise_scores = FileIO.readFile_lineList(scoresFile);
        Map<String, double[]> pairwise_scoreDict = new HashMap<>();
        for(String line : ll_pairwise_scores){
            String[] lineParts = line.split(",");
            double[] scores = new double[4];
            for(int i=1; i<lineParts.length; i++)
                scores[i-1] = Math.exp(Double.parseDouble(lineParts[i]));
            pairwise_scoreDict.put(lineParts[0], scores);
        }
        Map<String, String> pairwise_scoreDict_label = new HashMap<>();
        for(String key : pairwise_scoreDict.keySet()){
            double max = 0.0;
            int idx = 0;
            for(int i = 0; i<pairwise_scoreDict.get(key).length; i++){
                double score = pairwise_scoreDict.get(key)[i];
                if(score > max){
                    max = score;
                    idx = i;
                }
            }
            String label = "n";
            switch(idx){
                case 0: label = "n";
                    break;
                case 1: label = "c";
                    break;
                case 2: label = "b";
                    break;
                case 3: label = "p";
                    break;
            }
            pairwise_scoreDict_label.put(key, label);
        }

        Map<String, DoubleDict<String>> mentionPairTable = new HashMap<>();
        mentionPairTable.put("n|n", new DoubleDict<>());
        mentionPairTable.put("c|c", new DoubleDict<>());
        mentionPairTable.put("b|p", new DoubleDict<>());
        for(Document d : docSet){
            Set<String> subsetMentions = d.getSubsetMentions();
            List<Mention> mentionList = d.getMentionList();
            for(int i=0; i<mentionList.size(); i++) {
                Mention m1 = mentionList.get(i);
                if(m1.getChainID().equals("0"))
                    continue;

                for (int j = i + 1; j < mentionList.size(); j++) {
                    Mention m2 = mentionList.get(j);
                    if(m2.getChainID().equals("0"))
                        continue;

                    String id_ij = Document.getMentionPairStr(m1, m2);
                    String id_ji = Document.getMentionPairStr(m2, m1);

                    String pred_ij = "n";
                    if(pairwise_scoreDict_label.containsKey(id_ij))
                        pred_ij = pairwise_scoreDict_label.get(id_ij);
                    String pred_ji = "n";
                    if(pairwise_scoreDict_label.containsKey(id_ji))
                        pred_ji = pairwise_scoreDict_label.get(id_ji);
                    String pred = StringUtil.getAlphabetizedPair(pred_ij, pred_ji);

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

    /**Used to generate the histograms for use as one-hot vectors;
     * this list consists of prepositions adjacent to mentions
     *
     * @param docSet
     */
    public static void export_prepositions(Collection<Document> docSet)
    {
        DoubleDict<String> prepHist = new DoubleDict<>();
        DoubleDict<String> prepPairHist = new DoubleDict<>();
        for(Document d : docSet){
            for(Caption c : d.getCaptionList()){
                for(int i=0; i<c.getMentionList().size(); i++){
                    List<Chunk> chunkList_i = c.getMentionList().get(i).getChunkList();
                    Chunk left_i = null, right_i = null;
                    if(!chunkList_i.isEmpty()){
                        left_i = c.getLeftNeighbor(chunkList_i.get(0));
                        right_i = c.getRightNeighbor(chunkList_i.get(chunkList_i.size()-1));
                    }

                    if(left_i != null && left_i.getChunkType().equals("PP"))
                        prepHist.increment(left_i.toString().toLowerCase());
                    if(right_i != null && right_i.getChunkType().equals("PP"))
                        prepHist.increment(right_i.toString().toLowerCase());

                    for(int j=i+1; j<c.getMentionList().size(); j++){
                        List<Chunk> chunkList_j = c.getMentionList().get(j).getChunkList();
                        Chunk left_j = null, right_j = null;
                        if(!chunkList_j.isEmpty()){
                            left_j = c.getLeftNeighbor(chunkList_j.get(0));
                            right_j = c.getRightNeighbor(chunkList_j.get(chunkList_j.size()-1));
                        }

                        if(left_i != null && left_i.getChunkType().equals("PP") &&
                           left_j != null && left_j.getChunkType().equals("PP")){
                            String left_ij = left_i.toString() + "|" + left_j.toString();
                            String left_ji = left_j.toString() + "|" + left_i.toString();
                            prepPairHist.increment(left_ij.toLowerCase());
                            prepPairHist.increment(left_ji.toLowerCase());
                        }

                        if(right_i != null && right_i.getChunkType().equals("PP") &&
                           right_j != null && right_j.getChunkType().equals("PP")){
                            String right_ij = right_i.toString() + "|" + right_j.toString();
                            String right_ji = right_j.toString() + "|" + right_i.toString();
                            prepPairHist.increment(right_ij.toLowerCase());
                            prepPairHist.increment(right_ji.toLowerCase());
                        }
                    }
                }
            }
        }
        FileIO.writeFile(prepHist, "hist_preposition", "csv", false);
        FileIO.writeFile(prepPairHist, "hist_prepPair", "csv", false);
    }

    /**Used to generate the histograms for use as one-hot vectors;
     * this list consists of mention modifiers
     *
     * @param docSet
     */
    public static void export_modifiers(Collection<Document> docSet)
    {
        DoubleDict<String> modHist_numeric = new DoubleDict<>();
        DoubleDict<String> modHist_other = new DoubleDict<>();
        DoubleDict<String> modPairHist_numeric = new DoubleDict<>();
        DoubleDict<String> modPairHist_other = new DoubleDict<>();
        for(Document d : docSet){
            List<Mention> mentions = d.getMentionList();
            for(int i=0; i<mentions.size(); i++){
                Mention m1 = mentions.get(i);
                String[] mods_1 = m1.getModifiers();
                String numericMod_1 = mods_1[0];
                String mod_1 = mods_1[1];

                if(!numericMod_1.isEmpty())
                    modHist_numeric.increment(numericMod_1);
                if(!mod_1.isEmpty())
                    modHist_other.increment(mod_1);

                for(int j=i+1; j<mentions.size(); j++){
                    Mention m2 = mentions.get(j);
                    String[] mods_2 = m2.getModifiers();
                    String numericMod_2 = mods_2[0];
                    String mod_2 = mods_2[1];

                    if(!numericMod_2.isEmpty())
                        modHist_numeric.increment(numericMod_2);
                    if(!mod_2.isEmpty())
                        modHist_other.increment(mod_2);

                    if(!numericMod_1.isEmpty() || !numericMod_2.isEmpty()) {
                        modPairHist_numeric.increment(
                                StringUtil.getAlphabetizedPair(numericMod_1,
                                        numericMod_2));
                    }
                    if(!mod_1.isEmpty() || !mod_2.isEmpty()){
                        modPairHist_other.increment(mod_1 + "|" + mod_2);
                        modPairHist_other.increment(mod_2 + "|" + mod_1);
                    }
                }
            }
        }

        Logger.log("Writing files");
        FileIO.writeFile(modHist_numeric, "hist_numericModifier", "csv", false);
        FileIO.writeFile(modPairHist_numeric, "hist_numericModifierPair", "csv", false);
        FileIO.writeFile(modHist_other, "hist_modifier", "csv", false);
        FileIO.writeFile(modPairHist_other, "hist_modifierPair", "csv", false);
    }

    /**Used to generate the histograms for use as one-hot vectors;
     * this list contains hypernym roots
     *
     * @param docSet
     */
    public static void export_hypernyms(Collection<Document> docSet)
    {
        WordnetUtil wnUtil = new WordnetUtil(Overlord.wordnetDir);
        DoubleDict<String> hypDict = new DoubleDict<>();
        DoubleDict<String> hypPairDict = new DoubleDict<>();
        Map<String, Set<String>> lemmaHypDict = new HashMap<>();
        for(Document d : docSet) {
            List<Mention> mentions = d.getMentionList();

            for (int i = 0; i < mentions.size(); i++) {
                Mention m_i = mentions.get(i);
                String lem_i = m_i.getHead().getLemma().toLowerCase();
                if (!lemmaHypDict.containsKey(lem_i)) {
                    Set<String> boh = wnUtil.getBagOfHypernyms(lem_i);
                    if (!boh.isEmpty())
                        lemmaHypDict.put(lem_i, boh);
                }

                Set<String> boh_i = new HashSet<>();
                if(lemmaHypDict.containsKey(lem_i))
                    boh_i = lemmaHypDict.get(lem_i);

                for (String h : boh_i)
                    hypDict.increment(h);

                for (int j = i + 1; j < mentions.size(); j++) {
                    Mention m_j = mentions.get(j);
                    String lem_j = m_j.getHead().getLemma().toLowerCase();

                    if (!lemmaHypDict.containsKey(lem_j)) {
                        Set<String> boh = wnUtil.getBagOfHypernyms(lem_j);
                        if (!boh.isEmpty())
                            lemmaHypDict.put(lem_j, boh);
                    }

                    Set<String> boh_j = new HashSet<>();
                    if(lemmaHypDict.containsKey(lem_j))
                        boh_j = lemmaHypDict.get(lem_j);

                    if (boh_i.isEmpty())
                        boh_i.add("");
                    if (boh_j.isEmpty())
                        boh_j.add("");
                    for (String h_i : boh_i)
                        for (String h_j : boh_j)
                            hypPairDict.increment(StringUtil.getAlphabetizedPair(h_i, h_j));
                }
            }
        }
        FileIO.writeFile(hypDict, "hist_hypernym", "csv", false);
        FileIO.writeFile(hypPairDict, "hist_hypernymPair", "csv", false);
    }

    public static void export_nonvisuals(Collection<Document> docSet)
    {
        DoubleDict<String> nonvisHist = new DoubleDict<>();
        for(Document d : docSet){
            for(Mention m : d.getMentionList()){
                if(m.getChainID().equals("0") && m.getPronounType() == Mention.PRONOUN_TYPE.NONE){
                    String lemma = m.getHead().getLemma().toLowerCase();
                    if(StringUtil.hasAlphaNum(lemma))
                        nonvisHist.increment(lemma);
                }
            }
        }
        FileIO.writeFile(nonvisHist, "hist_nonvisual", "csv", false);
    }

    public static void export_typePairFreq(Collection<Document> docSet)
    {
        /*Create a lex/pronoun type matrix for pairs of mentions*/
        List<String> lexAndPronomTypes = new ArrayList<>();
        lexAndPronomTypes.add("animals"); lexAndPronomTypes.add("bodyparts");
        lexAndPronomTypes.add("clothing"); lexAndPronomTypes.add("colors");
        lexAndPronomTypes.add("instruments"); lexAndPronomTypes.add("other");
        lexAndPronomTypes.add("people"); lexAndPronomTypes.add("scene");
        lexAndPronomTypes.add("vehicles"); lexAndPronomTypes.add("animate");
        lexAndPronomTypes.add("semi"); lexAndPronomTypes.add("other_pro");
        Collections.sort(lexAndPronomTypes);

        //Count how often a pair of labels (nn, cc, bp, pb) appears with type pairs ij / ji
        //NOTE: keep in mind that we're only storing the upper right half of this matrix
        Logger.log("Counting type pair frequencies");
        double[][][] typePairFreqs = new double[lexAndPronomTypes.size()][lexAndPronomTypes.size()][4];
        for(int i=0; i<lexAndPronomTypes.size(); i++) {
            typePairFreqs[i] = new double[lexAndPronomTypes.size()][4];
            for(int j=0; j<lexAndPronomTypes.size(); j++)
                typePairFreqs[i][j] = new double[]{0.0, 0.0, 0.0, 0.0};
        }
        for(Document d : docSet) {
            Set<String> subsetMentions = d.getSubsetMentions();

            List<Mention> mentions = d.getMentionList();
            for (int i = 0; i < mentions.size(); i++) {
                Mention m_i = mentions.get(i);
                //Ignore nonvisual mentions and training pronouns
                //that haven't been reviewed
                if (m_i.getChainID().equals("0") ||
                        m_i.getPronounType() != Mention.PRONOUN_TYPE.NONE && !d.reviewed)
                    continue;

                for (int j = 0; j < mentions.size(); j++) {
                    Mention m_j = mentions.get(j);
                    if (m_j.getChainID().equals("0") ||
                            m_j.getPronounType() != Mention.PRONOUN_TYPE.NONE && !d.reviewed)
                        continue;

                    int label = 0;
                    if (m_i.getChainID().equals(m_j.getChainID()))
                        label = 1;
                    else if (subsetMentions.contains(Document.getMentionPairStr(m_i, m_j)))
                        label = 2;
                    else if(subsetMentions.contains(Document.getMentionPairStr(m_j, m_i)))
                        label = 3;

                    //Find this mention's pronoun type (if pronominal) or lexical types (if not)
                    String type_i = ClassifyUtil.getTypeCostLabel(m_i);
                    String[] types_i = type_i.split("/");
                    String type_j = ClassifyUtil.getTypeCostLabel(m_j);
                    String[] types_j = type_j.split("/");

                    //in cases where a mention has multiple types, we'll
                    //count it fractionally in accordance with how many types it has
                    double norm = types_i.length * types_j.length;
                    for (String t_i : types_i) {
                        for (String t_j : types_j) {
                            int tIdx_i = lexAndPronomTypes.indexOf(t_i);
                            int tIdx_j = lexAndPronomTypes.indexOf(t_j);
                            typePairFreqs[tIdx_i][tIdx_j][label] += 1.0 / norm;
                        }
                    }
                }
            }
        }

        //Now that we have frequencies for each unordered ij with valid label pair y,y'
        //we want to normalize them such that \sum_{y,y' \in nn, cc, bp, pb} t_{ij}(y,y') = 1
        double[][][] typeProbs_log = new double[lexAndPronomTypes.size()][lexAndPronomTypes.size()][4];
        for(int i=0; i<lexAndPronomTypes.size(); i++){
            for(int j=0; j<lexAndPronomTypes.size(); j++) {
                double norm = Math.log(StatisticalUtil.getSum(typePairFreqs[i][j]));
                for(int y=0; y<4; y++) {
                    double freq = typePairFreqs[i][j][y];
                    if(freq == 0)
                        typeProbs_log[i][j][y] = Math.log(Double.MIN_VALUE);
                    else
                        typeProbs_log[i][j][y] = Math.log(freq) - norm;
                }
            }
        }

        //Now that we have those probabilities, we divide each by 2, corresponding to
        //each label's individual probability
        List<String> ll_probs = new ArrayList<>();
        List<String> ll_logProbs = new ArrayList<>();
        for(int i=0; i<lexAndPronomTypes.size(); i++){
            for(int j=0; j<lexAndPronomTypes.size(); j++) {
                String id_ij = lexAndPronomTypes.get(i) + "|" + lexAndPronomTypes.get(j);
                double[] logProbs = {0.0, 0.0, 0.0, 0.0};
                logProbs[0] = typeProbs_log[i][j][0];
                logProbs[1] = typeProbs_log[i][j][1];
                logProbs[2] = typeProbs_log[i][j][2];
                logProbs[3] = typeProbs_log[i][j][3];

                double[] probs = {0.0, 0.0, 0.0, 0.0};
                for(int y=0; y<4; y++)
                    probs[y] = Math.exp(logProbs[y]);

                //Add the probabilities to the appropriate line lists
                ll_probs.add(id_ij + "," + probs[0] + "," + probs[1] +
                        "," + probs[2] + "," + probs[3]);
                ll_logProbs.add(id_ij + "," + logProbs[0] + "," + logProbs[1] +
                        "," + logProbs[2] + "," + logProbs[3]);
            }
        }

        FileIO.writeFile(ll_probs, "hist_typePairProb", "csv", false);
        FileIO.writeFile(ll_logProbs, "hist_typePairLogProb", "csv", false);
    }

    /**Converts the bounding box feats in the VGG box file (~10G) to
     * individual doc-level .feats format
     *
     * @param docSet
     */
    public static void export_convertVGGBoxFeats(Collection<Document> docSet)
    {
        Logger.log("Reading box feats into memory");
        Map<String, String> dimFeatsDict = new HashMap<>();
        try{
            BufferedReader br = new BufferedReader(
                    new InputStreamReader(
                            new FileInputStream("/home/ccervan2/source/data/Flickr30kEntities_v1/flickr30kEntities_boxes.feats")));
            String nextLine = br.readLine();
            nextLine = br.readLine();
            while(nextLine != null){
                String[] lineArr = nextLine.split(",");
                String boxDims = lineArr[0] + "," + lineArr[1] + "," +
                        lineArr[2] + "," + lineArr[3] + "," + lineArr[4];
                dimFeatsDict.put(boxDims, lineArr[5]);
                nextLine = br.readLine();
            }
        } catch(Exception ex){
            Logger.log(ex);
        }

        Logger.log("Processing box feats");
        int docIdx = 0;
        for (Document d : docSet){
            Set<FeatureVector> fvSet = new HashSet<>();
            for(BoundingBox b : d.getBoundingBoxSet()) {
                String boxDims = d.getID().replace(".jpg", "") + "," +
                        b.getXMin() + "," + b.getYMin() +
                        "," + b.getXMax() + "," + b.getYMax();
                if(dimFeatsDict.containsKey(boxDims)) {
                    List<Double> vals = new ArrayList<>();
                    for(String v : dimFeatsDict.get(boxDims).split("\\|"))
                        vals.add(Double.parseDouble(v));
                    fvSet.add(new FeatureVector(vals, 0, b.getUniqueID()));
                } else {
                    System.out.println("MISSING BOX FEATS: " + b.getUniqueID());
                }
            }

            List<String> ll_fv = new ArrayList<>();
            fvSet.forEach(fv -> ll_fv.add(fv.toString()));
            FileIO.writeFile(ll_fv, boxFeatureDir + d.getID().replace(".jpg", ""), "feats", false);
            docIdx++;
            Logger.logStatus("%.2f%% complete", 100.0 * (double)docIdx / docSet.size());
        }
    }

    /**Exports data for use in Bryan's CCA affinity model
     *
     * @param docSet
     * @param dataSplit
     */
    public static void export_bryanPreproc(Collection<Document> docSet, String dataSplit)
    {
        //Read bounding box filenames from the box feature dir
        Set<String> boxFiles = new HashSet<>();
        File boxDir = new File(Overlord.boxFeatureDir);
        for(File f : boxDir.listFiles())
            if(f.isFile())
                boxFiles.add(f.getName().replace(".feats", ""));

        //Store a mapping of [docID -> [mention -> [bounding boxes] ] ]
        Map<String, Map<Mention, Set<BoundingBox>>>
                mentionBoxesDict = new HashMap<>();
        for(Document d : docSet){
            if(boxFiles.contains(d.getID().replace(".jpg", ""))){
                for(Mention m : d.getMentionList()){
                    String normText = m.toString().toLowerCase().trim();
                    if(!mentionBoxesDict.containsKey(normText))
                        mentionBoxesDict.put(normText, new HashMap<>());
                    mentionBoxesDict.get(normText).put(m, d.getBoxSetForMention(m));
                }
            } else {
                System.out.println("ERROR: found no box feats for " + d.getID());
            }
        }

        //store the in-order lists of box feats and text strings
        List<String> ll_img = new ArrayList<>();
        List<String> ll_txt = new ArrayList<>();
        List<String> ll_ids = new ArrayList<>();
        if(dataSplit.equals("train")){
            //Randomly sample 10 bounding boxes from those that have boxes
            Map<String, Set<BoundingBox>> mentionBoxesDict_sample =
                    new HashMap<>();
            for(String normText : mentionBoxesDict.keySet()){
                List<BoundingBox> boxList = new ArrayList<>();
                for(Mention m : mentionBoxesDict.get(normText).keySet())
                    boxList.addAll(mentionBoxesDict.get(normText).get(m));
                Collections.shuffle(boxList);
                if(!boxList.isEmpty()) {
                    mentionBoxesDict_sample.put(normText,
                            new HashSet<>(boxList.subList(0,
                                    Math.min(boxList.size() - 1, 10))));
                }
            }

            //store the in-order lists of box feats and text feats
            int text_idx = 0;
            for(String normText : mentionBoxesDict_sample.keySet()) {
                for(BoundingBox b : mentionBoxesDict_sample.get(normText)){
                    ll_txt.add(normText);

                    //Since we don't know which im—e word's box will come from
                    //a-priori, open the doc's file at each box (less efficient, but necessary)
                    List<Double> imgFeats = null;
                    try{
                        BufferedReader br = new BufferedReader(new InputStreamReader(
                                new FileInputStream(Overlord.boxFeatureDir +
                                        b.getDocID().replace(".jpg", ".feats"))));
                        String nextLine = br.readLine();
                        while(nextLine != null && imgFeats == null){
                            String fvID = nextLine.split(" # ")[1];
                            if(fvID.equals(b.getUniqueID())){
                                FeatureVector fv = FeatureVector.parseFeatureVector(nextLine);
                                imgFeats = new ArrayList<>();
                                for(int i=1; i<=4096; i++)
                                    imgFeats.add(fv.getFeatureValue(i));
                            }
                            nextLine = br.readLine();
                        }
                        br.close();
                    } catch(Exception ex) {Logger.log(ex);}

                    //add the features to the img file
                    ll_img.add(StringUtil.listToString(imgFeats, ","));
                }
                text_idx++;
                Logger.logStatus("Completed %.2f%%", 100.0 * text_idx / mentionBoxesDict_sample.size());
            }
        } else {
            int docIdx = 0;
            for(Document d : docSet){
                //read box features for this file
                Map<String, List<Double>> boxFeatDict = new HashMap<>();
                List<String> ll_boxes = FileIO.readFile_lineList(Overlord.boxFeatureDir +
                        d.getID().replace(".jpg", ".feats"));
                for(String featStr : ll_boxes){
                    FeatureVector fv = FeatureVector.parseFeatureVector(featStr);
                    List<Double> boxFeats = new ArrayList<>();
                    for(int i=1; i<=4096; i++)
                        boxFeats.add(fv.getFeatureValue(i));
                    boxFeatDict.put(fv.comments, boxFeats);
                }

                //for each mention / box pair
                for(Mention m : d.getMentionList()){
                    String normText = m.toString().toLowerCase().trim();
                    for(BoundingBox b : d.getBoundingBoxSet()){
                        ll_img.add(StringUtil.listToString(boxFeatDict.get(b.getUniqueID()), ","));
                        ll_txt.add(normText);
                        ll_ids.add(m.getUniqueID() + "|" + b.getUniqueID());
                    }
                }
                docIdx++;
                Logger.logStatus("Completed %.2f%%", 100.0 * docIdx / docSet.size());
            }
        }

        FileIO.writeFile(ll_img, Overlord.dataPath + "feats/img_" + dataSplit, "csv", false);
        FileIO.writeFile(ll_txt, Overlord.dataPath + "feats/txt_" + dataSplit, "txt", false);
        if(!ll_ids.isEmpty())
            FileIO.writeFile(ll_ids, Overlord.dataPath + "feats/ids_" + dataSplit, "txt", false);
    }

    /**Exports a label and type file, necessary for learning the curves (in iclccaAffinity.py;
     * part of ImageCaptionLearn_py) based on Bryan's CCA scores
     *
     * @param docSet
     * @param dataSplit
     */
    public static void export_bryanPostproc(Collection<Document> docSet, String dataSplit, String idFile)
    {
        //For each box/mention pair, associate it with its label
        //and while we're looping, grab lexical types too
        Map<String, Integer> labelDict = new HashMap<>();
        List<String> ll_types = new ArrayList<>();
        for(Document d : docSet){
            for(Mention m : d.getMentionList()){
                ll_types.add(m.getUniqueID() + "," + m.getLexicalType());

                Set<BoundingBox> boxSet = d.getBoxSetForMention(m);
                for(BoundingBox b : d.getBoundingBoxSet()){
                    int label = boxSet.contains(b) ? 1 : 0;
                    labelDict.put(m.getUniqueID() + "|" + b.getUniqueID(), label);
                }
            }
        }
        FileIO.writeFile(ll_types, "cca_types_" + dataSplit, "csv", false);

        // To make doubly sure that ID order is preserved, read the file
        // for IDs (rather than regenerating from docSet)
        List<String> ll_labels = new ArrayList<>();
        for(String id : FileIO.readFile_lineList(idFile))
            if(labelDict.containsKey(id))
                ll_labels.add(String.valueOf(labelDict.get(id)));
        FileIO.writeFile(ll_labels, "cca_labels_" + dataSplit, "txt", false);
    }

    public static void buildImageCaptionDB(String corefFile, String releaseDir,
                                           String commentsFile, String crossValFile,
                                           String reviewedFile, String dbName)
    {
        Collection<Document> docSet = getJointDocumentSet(corefFile, releaseDir,
                commentsFile, crossValFile, reviewedFile);

        Logger.log("Uploading everything to the DB");
        DBConnector conn = new DBConnector(dbName);
        try{
            DocumentLoader.populateDocumentDB(conn, docSet, 100000, 1);
        } catch(Exception ex){
            utilities.Logger.log(ex);
        }
    }

    public static void buildImageCaptionDB(String corefFile, String releaseDir, String commentsFile,
                                           String crossvalFile, String reviewedFile, String host,
                                           String user, String password, String name)
    {
        Collection<Document> docSet = getJointDocumentSet(corefFile, releaseDir,
                commentsFile, crossvalFile, reviewedFile);

        Logger.log("Uploading everything to the DB");
        DBConnector conn = new DBConnector(host, user, password, name);
        try{
            DocumentLoader.populateDocumentDB(conn, docSet, 100000, 1);
        } catch(Exception ex){
            utilities.Logger.log(ex);
        }
    }

    /**Returns the document set using the combination of the documents in the specified
     * coref file, 30kEntities release dir, and
     *
     * @param corefFile
     * @param releaseDir
     * @return
     */
    private static Collection<Document> getJointDocumentSet(String corefFile,
                    String releaseDir, String commentsFile, String crossvalFile,
                    String reviewedFile)
    {
        Logger.log("Loading documents from coref file");
        Collection<Document> docSet_coref = DocumentLoader.getDocumentSet(
                corefFile,
                Overlord.lexPath, Overlord.resourcesDir);
        Map<String, Document> docDict_coref = new HashMap<>();
        docSet_coref.forEach(d -> docDict_coref.put(d.getID(), d));

        Logger.log("Loading documents from flickr30kEntities file");
        Collection<Document> docSet_flickr =
                DocumentLoader.getDocumentSet(releaseDir,
                        Overlord.resourcesDir);
        Map<String, Document> docDict_flickr = new HashMap<>();
        docSet_flickr.forEach(d -> docDict_flickr.put(d.getID(), d));

        Logger.log("Merging documents");
        for(String docID : docDict_coref.keySet())
            docDict_coref.get(docID).loadBoxesFromDocument(docDict_flickr.get(docID));

        Logger.log("Adding annotation comments");
        String[][] commentsTable = FileIO.readFile_table(commentsFile);
        for(String[] row : commentsTable)
                docDict_coref.get(row[0] + ".jpg").comments = row[1];

        Logger.log("Adding cross val flags");
        String[][] crossValTable = FileIO.readFile_table(crossvalFile);
        for(String[] row : crossValTable)
            docDict_coref.get(row[0] + ".jpg").crossVal = Integer.parseInt(row[1]);

        Logger.log("Adding reviewed flag");
        Set<String> reviewedImgs = new HashSet<>(FileIO.readFile_lineList(reviewedFile));
        for(String imgID : reviewedImgs)
            docDict_coref.get(imgID + ".jpg").reviewed = true;

        return docDict_coref.values();
    }

    public static void exportDatasetLists()
    {
        Logger.setVerbose();
        Logger.log("Loading documents from DB");
        DBConnector conn = new DBConnector(dbPath);
        Collection<Document> docSet = DocumentLoader.getDocumentSet(conn, 1);

        Logger.log("Building lists");
        DoubleDict<String> detDict = new DoubleDict<>();
        for(Document d : docSet)
            for(Caption c : d.getCaptionList())
                for(Token t : c.getTokenList())
                    if(t.getPosTag().equals("DT"))
                        detDict.increment(t.toString().toLowerCase());
        DoubleDict<String> typePairDict = new DoubleDict<>();
        DoubleDict<String> leftPairDict = new DoubleDict<>();
        DoubleDict<String> rightPairDict = new DoubleDict<>();
        DoubleDict<String> headPairDict = new DoubleDict<>();
        DoubleDict<String> lemmaPairDict = new DoubleDict<>();
        DoubleDict<String> subjOfPairDict = new DoubleDict<>();
        DoubleDict<String> objOfPairDict = new DoubleDict<>();
        DoubleDict<String> typePairDict_ordered = new DoubleDict<>();
        DoubleDict<String> leftPairDict_ordered = new DoubleDict<>();
        DoubleDict<String> rightPairDict_ordered = new DoubleDict<>();
        DoubleDict<String> headPairDict_ordered = new DoubleDict<>();
        DoubleDict<String> lemmaPairDict_ordered = new DoubleDict<>();
        DoubleDict<String> subjOfPairDict_ordered = new DoubleDict<>();
        DoubleDict<String> objOfPairDict_ordered = new DoubleDict<>();
        DoubleDict<String> headDict = new DoubleDict<>();
        DoubleDict<String> typeDict = new DoubleDict<>();
        DoubleDict<String> leftDict = new DoubleDict<>();
        DoubleDict<String> rightDict = new DoubleDict<>();
        DoubleDict<String> subjOfDict = new DoubleDict<>();
        DoubleDict<String> objOfDict = new DoubleDict<>();

        int totalPairs = 0;
        for(Document d : docSet) {
            int numMentions = d.getMentionList().size();
            totalPairs += (0.5 * numMentions * (numMentions -1));
        }
        int pairIdx = 0;

        for(Document d : docSet){
            List<Mention> mentionList = d.getMentionList();
            for(int i=0; i<mentionList.size(); i++){
                Mention m1 = mentionList.get(i);
                Caption c1 = d.getCaption(m1.getCaptionIdx());

                String head_1 = m1.getHead().toString().toLowerCase().replace(",", "");
                String lemma_1 = m1.getHead().getLemma().toLowerCase().replace(",", "");
                String type_1 = m1.getLexicalType().toLowerCase().replace(",", "");
                Chunk subjOf_1 = c1.getSubjectOf(m1);
                Chunk objOf_1 = c1.getObjectOf(m1);

                int leftChunkIdx_1 = -1;
                int rightChunkIdx_1 = Integer.MAX_VALUE;
                if(!m1.getChunkList().isEmpty()){
                    leftChunkIdx_1 = m1.getChunkList().get(0).getIdx() - 1;
                    rightChunkIdx_1 = m1.getChunkList().get(m1.getChunkList().size()-1).getIdx() + 1;
                }
                String left_1 = "START";
                if(leftChunkIdx_1 >= 0)
                    left_1 = c1.getChunkList().get(leftChunkIdx_1).getChunkType();
                String right_1 = "END";
                if(rightChunkIdx_1 < c1.getChunkList().size())
                    right_1 = c1.getChunkList().get(rightChunkIdx_1).getChunkType();

                headDict.increment(head_1);
                typeDict.increment(type_1);
                leftDict.increment(left_1);
                rightDict.increment(right_1);

                String subjOfStr_1 = null;
                String objOfStr_1 = null;
                if(subjOf_1 != null){
                    subjOfStr_1 = subjOf_1.getTokenList().get(subjOf_1.getTokenList().size()-1).toString();
                    subjOfStr_1 = subjOfStr_1.toLowerCase().replace(",", "");
                    subjOfDict.increment(subjOfStr_1);
                }
                if(objOf_1 != null){
                    objOfStr_1 = objOf_1.getTokenList().get(objOf_1.getTokenList().size()-1).toString();
                    objOfStr_1 = objOfStr_1.toLowerCase().replace(",", "");
                    objOfDict.increment(objOfStr_1);
                }

                for(int j=i+1; j<mentionList.size(); j++){
                    Mention m2 = mentionList.get(j);

                    Logger.logStatus("Computed %d (%.2f%%) mention pairs",
                            pairIdx, 100.0 * pairIdx / totalPairs);
                    pairIdx++;


                    String type_2 = m2.getLexicalType().toLowerCase().replace(",", "");
                    String head_2 = m2.getHead().toString().toLowerCase().replace(",", "");
                    String lemma_2 = m2.getHead().getLemma().toLowerCase().replace(",", "");

                    typePairDict.increment(StringUtil.getAlphabetizedPair(type_1, type_2));
                    typePairDict_ordered.increment(type_1 + "|" + type_2);
                    typePairDict_ordered.increment(type_2 + "|" + type_1);
                    headPairDict.increment(StringUtil.getAlphabetizedPair(head_1, head_2));
                    headPairDict_ordered.increment(head_1 + "|" + head_2);
                    headPairDict_ordered.increment(head_2 + "|" + head_1);
                    lemmaPairDict.increment(StringUtil.getAlphabetizedPair(lemma_1, lemma_2));
                    lemmaPairDict_ordered.increment(lemma_1 + "|" + lemma_2);
                    lemmaPairDict_ordered.increment(lemma_2 + "|" + lemma_1);

                    Caption c2 = d.getCaption(m2.getCaptionIdx());
                    Chunk subjOf_2 = c2.getSubjectOf(m2);
                    Chunk objOf_2 = c2.getObjectOf(m2);

                    String subjOfStr_2 = null;
                    String objOfStr_2 = null;
                    if(subjOf_2 != null){
                        subjOfStr_2 = subjOf_2.getTokenList().get(subjOf_2.getTokenList().size()-1).toString();
                        subjOfStr_2 = subjOfStr_2.toLowerCase().replace(",", "");
                    }
                    if(objOf_2 != null){
                        objOfStr_2 = objOf_2.getTokenList().get(objOf_2.getTokenList().size()-1).toString();
                        objOfStr_2 = objOfStr_2.toLowerCase().replace(",", "");
                    }

                    int leftChunkIdx_2 = -1;
                    int rightChunkIdx_2 = Integer.MAX_VALUE;
                    if(!m2.getChunkList().isEmpty()){
                        leftChunkIdx_2 = m2.getChunkList().get(0).getIdx() - 1;
                        rightChunkIdx_2 = m2.getChunkList().get(m2.getChunkList().size()-1).getIdx() + 1;
                    }
                    String left_2 = "START";
                    if(leftChunkIdx_2 >= 0)
                        left_2 = c2.getChunkList().get(leftChunkIdx_2).getChunkType();
                    String right_2 = "END";
                    if(rightChunkIdx_2 < c2.getChunkList().size())
                        right_2 = c2.getChunkList().get(rightChunkIdx_2).getChunkType();
                    leftPairDict.increment(StringUtil.getAlphabetizedPair(left_1, left_2));
                    rightPairDict.increment(StringUtil.getAlphabetizedPair(right_1, right_2));
                    leftPairDict_ordered.increment(left_1 + "|" + left_2);
                    leftPairDict_ordered.increment(left_2 + "|" + left_1);
                    rightPairDict_ordered.increment(right_1 + "|" + right_2);
                    rightPairDict_ordered.increment(right_2 + "|" + right_1);
                    if(subjOfStr_1 != null && subjOfStr_2 != null){
                        subjOfPairDict.increment(StringUtil.getAlphabetizedPair(subjOfStr_1, subjOfStr_2));
                        subjOfPairDict_ordered.increment(subjOfStr_1 + "|" + subjOfStr_2);
                        subjOfPairDict_ordered.increment(subjOfStr_2 + "|" + subjOfStr_1);
                    }
                    if(objOfStr_1 != null && objOfStr_2 != null){
                        objOfPairDict.increment(StringUtil.getAlphabetizedPair(objOfStr_1, objOfStr_2));
                        objOfPairDict_ordered.increment(objOfStr_1 + "|" + objOfStr_2);
                        objOfPairDict_ordered.increment(objOfStr_2 + "|" + objOfStr_1);
                    }
                }
            }
        }
        Logger.log("Writing single files");
        FileIO.writeFile(detDict, "hist_det", "csv", false);
        FileIO.writeFile(headDict, "hist_head", "csv", false);
        FileIO.writeFile(typeDict, "hist_type", "csv", false);
        FileIO.writeFile(leftDict, "hist_left", "csv", false);
        FileIO.writeFile(rightDict, "hist_right", "csv", false);
        FileIO.writeFile(subjOfDict, "hist_subjOf", "csv", false);
        FileIO.writeFile(objOfDict, "hist_objOf", "csv", false);

        Logger.log("Writing unordered pair files");
        FileIO.writeFile(typePairDict, "hist_typePair", "csv", false);
        FileIO.writeFile(leftPairDict, "hist_leftPair", "csv", false);
        FileIO.writeFile(rightPairDict, "hist_rightPair", "csv", false);
        FileIO.writeFile(headPairDict, "hist_headPair", "csv", false);
        FileIO.writeFile(lemmaPairDict, "hist_lemmaPair", "csv", false);
        FileIO.writeFile(subjOfPairDict, "hist_subjOfPair", "csv", false);
        FileIO.writeFile(objOfPairDict, "hist_objOfPair", "csv", false);

        Logger.log("Writing ordered pair files");
        FileIO.writeFile(typePairDict_ordered, "hist_typePair_ordered", "csv", false);
        FileIO.writeFile(leftPairDict_ordered, "hist_leftPair_ordered", "csv", false);
        FileIO.writeFile(rightPairDict_ordered, "hist_rightPair_ordered", "csv", false);
        FileIO.writeFile(headPairDict_ordered, "hist_headPair_ordered", "csv", false);
        FileIO.writeFile(lemmaPairDict_ordered, "hist_lemmaPair_ordered", "csv", false);
        FileIO.writeFile(subjOfPairDict_ordered, "hist_subjOfPair_ordered", "csv", false);
        FileIO.writeFile(objOfPairDict_ordered, "hist_objOfPair_ordered", "csv", false);
    }

    /**Returns the mentions in given Document d with predicted chain IDs from
     * a combination of given pairwise coref score dict pcDict and given
     * pronominal coreferent mention set pronomMentions
     *
     * @param d
     * @param pcDict
     * @param pronomMentions
     * @return
     */
    private static Map<Mention, Integer> getMentionChainDict(Document d,
                    BinaryClassifierScoreDict pcDict, Set<Mention[]> pronomMentions)
    {
        Map<Mention, Integer> mentionChainDict = new HashMap<>();
        int chainIdx = 1;

        List<Mention> mentionList = d.getMentionList();
        for(int i=0; i<mentionList.size(); i++){
            Mention m1 = mentionList.get(i);
            if(m1.getChainID().equals("0"))
                continue;
            for(int j=i+1; j<mentionList.size(); j++){
                Mention m2 = mentionList.get(j);
                if(m2.getChainID().equals("0"))
                    continue;

                int pred = 0;

                //if we were given pairwise scores, set the pred label
                if(pcDict != null && pcDict.get(m1, m2) != null)
                    pred = pcDict.get(m1, m2) > 0 ? 1 : 0;

                //if this pair is in the pronom dict, enforce the link
                if(pronomMentions != null){
                    for(Mention[] pair : pronomMentions){
                        if(pair[0].equals(m1) && pair[1].equals(m2)){
                            pred = 1;
                            break;
                        }
                    }
                }

                if(pred == 1) {
                    if (!mentionChainDict.containsKey(m1) && !mentionChainDict.containsKey(m2)) {
                        mentionChainDict.put(m1, chainIdx);
                        mentionChainDict.put(m2, chainIdx);
                        chainIdx++;
                    } else if (!mentionChainDict.containsKey(m1)) {
                        mentionChainDict.put(m1, mentionChainDict.get(m2));
                    } else if (!mentionChainDict.containsKey(m2)) {
                        mentionChainDict.put(m2, mentionChainDict.get(m1));
                    } else {
                        Set<Mention> reassigSet = new HashSet<>();
                        for (Mention m : mentionChainDict.keySet())
                            if (mentionChainDict.get(m).equals(mentionChainDict.get(m2)))
                                reassigSet.add(m);
                        for (Mention m : reassigSet)
                            mentionChainDict.put(m, mentionChainDict.get(m1));
                    }
                } else {
                    if(!mentionChainDict.containsKey(m1)){
                        mentionChainDict.put(m1, chainIdx);
                        chainIdx++;
                    }
                    if(!mentionChainDict.containsKey(m2)){
                        mentionChainDict.put(m2, chainIdx);
                        chainIdx++;
                    }
                }
            }
        }
        return mentionChainDict;
    }

    public static Map<String, String> getLemmaTypeDict(boolean includePronouns)
    {
        String[] types = {"animals", "bodyparts", "clothing",
                "colors", "instruments", "people",
                "scene", "vehicles"};
        Map<String, Set<String>> lemmaTypeSetDict = new HashMap<>();
        for(String type : types){
            List<String> lineList = FileIO.readFile_lineList(Overlord.lexPath + type + ".txt");
            for(String lemma : lineList){
                if(!lemmaTypeSetDict.containsKey(lemma))
                    lemmaTypeSetDict.put(lemma, new HashSet<>());
                lemmaTypeSetDict.get(lemma).add(type);
            }
        }
        Map<String, String> lemmaTypeDict = new HashMap<>();
        for(String lemma : lemmaTypeSetDict.keySet())
            lemmaTypeDict.put(lemma, StringUtil.listToString(lemmaTypeSetDict.get(lemma), "/"));

        if(includePronouns){
            String[] animatePronouns = {"he", "she", "her", "him", "they",
                    "them", "himself", "herself",
                    "themselves", "who", "whom"};
            for(String pronoun : animatePronouns)
                lemmaTypeDict.put(pronoun, "animals/people");
        }

        Set<String> typeSet = new HashSet<>(lemmaTypeDict.values());
        System.out.println(StringUtil.listToString(typeSet, " | "));

        return lemmaTypeDict;
    }

    public static double getAttrOverlap(AttrStruct root1, AttrStruct root2, String attrName)
    {
        double overlap = 0.0;
        if(root1 != null && root2 != null){
            int overlapCount = AttrStruct.getAttributeOverlap(root1, root2, attrName);
            Collection<AttrStruct> attrs1 = root1.getAttribute_struct(attrName);
            Collection<AttrStruct> attrs2 = root2.getAttribute_struct(attrName);
            if(attrs1 != null && !attrs1.isEmpty() &&
                    attrs2 != null && !attrs2.isEmpty()){
                int count1 = 0;
                for(AttrStruct as : attrs1)
                    count1 += as.getNumAttributes();
                int count2 = 0;
                for(AttrStruct as : attrs2)
                    count2 += as.getNumAttributes();
                overlap = (double)overlapCount / (double)(count1 + count2 - overlapCount);
            }
        }
        return overlap;
    }

    /**Returns a mapping of comments->featureVector from a .feats
     * file for a given Document
     *
     * @param docID
     * @return
     */
    private static Map<String, FeatureVector>
        readFeatureFile(String featsFile, String docID) throws IOException
    {
        Map<String, FeatureVector> feats = new HashMap<>();
        BufferedReader br = new BufferedReader(
                new InputStreamReader(
                        new FileInputStream(featsFile)));
        String nextLine = br.readLine();
        while(nextLine != null) {
            if(nextLine.contains(docID)){
                FeatureVector fv = FeatureVector.parseFeatureVector(nextLine);
                feats.put(fv.comments, fv);
            }
            nextLine = br.readLine();
        }
        return feats;
    }

    /**CONLL 2000 chunking input format is
     *
     * token POS gold pred
     *
     */
    public static void exportAnnoDiffConllEval()
    {
        Collection<Document> docSet =
                DocumentLoader.getDocumentSet(new DBConnector(dbPath));
        Collection<Document> docSet_legacy =
                DocumentLoader.getDocumentSet(new DBConnector(Overlord.dbPath_legacy));
        Map<String, Document> docDict_legacy =
                new HashMap<>();
        for(Document d : docSet_legacy)
            docDict_legacy.put(d.getID(), d);

        Map<String, Integer> annoQueueDict = new HashMap<>();
        for(String[] row : FileIO.readFile_table(Overlord.resourcesDir + "annoQueue.csv"))
            annoQueueDict.put(row[0], Integer.parseInt(row[1]));

        List<String> ll = new ArrayList<>();
        int reviewedCount = 0;
        int sameCaps = 0;
        int diffCaps = 0;
        for(Document d : docSet){
            if(d.reviewed){
                reviewedCount++;
                Document d_legacy = docDict_legacy.get(d.getID());

                for(int capIdx=0; capIdx<d.getCaptionList().size(); capIdx++) {
                    Caption c = d.getCaption(capIdx);
                    Caption c_legacy = d_legacy.getCaption(capIdx);
                    if (c.getTokenList().size() != c_legacy.getTokenList().size()){
                        diffCaps++;
                    } else {
                        sameCaps++;
                        ll.addAll(c_legacy.toConllStrings(c.getTokenList()));

                        //add an empty line to the line list after each sentence
                        ll.add("");
                    }
                }
            }
        }
        Logger.log("Reviewed img count: %d", reviewedCount);
        Logger.log("Caption count: %d same; %d diff ", sameCaps, diffCaps);
        FileIO.writeFile(ll, "annoDiff", "conll", false);
    }

    public static void convertFeatsFile(String srcFile, String destDir)
    {

        List<String> featStrs = FileIO.readFile_lineList(srcFile);
        int numVectors = featStrs.size();
        Map<String, Set<FeatureVector>> fvDict = new HashMap<>();
        for(String fStr : featStrs){
            FeatureVector fv = FeatureVector.parseFeatureVector(fStr);
            String imgID = fv.comments.split(".jpg")[0].replace("doc:", "");
            if(!fvDict.containsKey(imgID))
                fvDict.put(imgID, new HashSet<>());
            fvDict.get(imgID).add(fv);
        }
        featStrs.clear(); featStrs = null;
        Logger.log("%d imgs", fvDict.keySet().size());

        int maxIdx = 0;
        for(Set<FeatureVector> fvSet : fvDict.values())
            for(FeatureVector fv : fvSet)
                for(int idx : fv.getFeatureIndices())
                    if(idx > maxIdx)
                        maxIdx = idx;

        int fileCount = 0;
        for(String imgID : fvDict.keySet()){
            List<String> outList = new ArrayList<>();
            for(FeatureVector fv : fvDict.get(imgID)){
                String line = fv.comments + "," + fv.label + ",";
                for(int i=0; i<maxIdx; i++){
                    Double val = fv.getFeatureValue(i);
                    if(val == null)
                        val = 0.0;
                    line += val;
                    if(i < maxIdx - 1)
                        line += ",";
                }
                outList.add(line);
            }
            fileCount++;
            FileIO.writeFile(outList, destDir + imgID, "feats", false);
            Logger.logStatus("Written %d docs (%.2f%%)", fileCount, 100.0 * (double)fileCount / numVectors);
        }
    }

    /**Assigns latex colors to the given set of IDs
     *
     * @param ids
     * @return
     */
    public static Map<String, String> getLatexColors(Collection<String> ids)
    {
        String[] colors = {"OliveGreen", "Blue", "Brown", "Red", "Plum", "BurntOrange",
                           "Aquamarine", "CadetBlue", "Gray", "GreenYellow", "Magenta",
                           "Salmon", "MidnightBlue", "Periwinkle", "Emerald", "RawSienna",
                           "Apricot", "Tan"};

        Map<String, String> colorDict = new HashMap<>();
        int i = 0; List<String> idList = new ArrayList<>(ids);
        while(i < idList.size() && i < colors.length) {
            colorDict.put(idList.get(i), colors[i]);
            i++;
        }
        return colorDict;
    }
}
