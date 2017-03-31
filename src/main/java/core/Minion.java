package core;

import learn.ClassifyUtil;
import learn.FeatureVector;
import nlptools.Chunker;
import nlptools.Tagger;
import nlptools.WordnetUtil;
import out.OutTable;
import structures.*;
import utilities.*;

import java.awt.geom.Area;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.InputStreamReader;
import java.util.*;
import java.util.function.Function;

import static core.Overlord.boxFeatureDir;

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

    public static void export_attachmentCases(Collection<Document> docSet)
    {
        Logger.log("Attaching clothing/bodyparts to agents");
        Map<Mention, AttrStruct> attributeDict =
                ClassifyUtil.attributeAttachment_agent(docSet);
        Map<String, Set<Mention[]>> attachedMentions = new HashMap<>();
        for(Mention m : attributeDict.keySet()){
            String capID = m.getDocID() + "#" + m.getCaptionIdx();
            AttrStruct as = attributeDict.get(m);
            for(Mention attrM : as.getAttributeMentions()){
                if(!attachedMentions.containsKey(capID))
                    attachedMentions.put(capID, new HashSet<>());

                Mention[] attrPair = new Mention[]{m, attrM};
                if(!Util.containsArr(attachedMentions.get(capID), attrPair))
                    attachedMentions.get(capID).add(attrPair);
            }
        }

        Logger.log("Computing statistics and collectiong examples");
        OutTable ot_cloth = new OutTable("doc_id", "cap_idx", "agents_in_cap",
                "cloth_in_cap", "agent", "clothing", "caption");
        OutTable ot_part = new OutTable("doc_id", "cap_idx", "agents_in_cap",
                "parts_in_cap", "agent", "bodyparts", "caption");
        OutTable otUnatt_cloth = new OutTable("doc_id", "cap_idx", "agents_in_cap",
                "cloth_in_cap", "clothing", "caption");
        OutTable otUnatt_part = new OutTable("doc_id", "cap_idx", "agents_in_cap",
                "parts_in_cap", "bodyparts", "caption");
        OutTable otMult_agents = new OutTable("doc_id", "cap_idx", "multi_type_agents", "caption");
        OutTable otMult_cloth = new OutTable("doc_id", "cap_idx", "multi_type_clothes", "caption");
        OutTable otMult_parts = new OutTable("doc_id", "cap_idx", "multi_type_parts", "caption");
        DoubleDict<String> hist_agentCloth = new DoubleDict<>();
        DoubleDict<String> hist_agentPart = new DoubleDict<>();
        DoubleDict<String> freqs = new DoubleDict<>();
        for(Document d : docSet){
            for(Caption c : d.getCaptionList()){
                //Get the total number of agents/cloth/parts in this caption
                Set<Mention> agents = new HashSet<>(), agents_mult = new HashSet<>();
                Set<Mention> clothes = new HashSet<>(), clothes_mult = new HashSet<>();
                Set<Mention> parts = new HashSet<>(), parts_mult = new HashSet<>();
                for(Mention m : c.getMentionList()){
                    if(m.getLexicalType().equals("people") ||
                       m.getLexicalType().equals("animals") ||
                       m.getPronounType().isAnimate()){
                        agents.add(m);
                    } else if(m.getLexicalType().equals("bodyparts")) {
                        parts.add(m);
                    } else if(m.getLexicalType().equals("clothing") ||
                              m.getLexicalType().equals("colors") ||
                              m.getLexicalType().equals("clothing/colors")){
                        clothes.add(m);
                    } else {
                        if(m.getLexicalType().contains("people") || m.getLexicalType().contains("animals"))
                            agents_mult.add(m);
                        else if(m.getLexicalType().contains("bodyparts"))
                            parts_mult.add(m);
                        else if(m.getLexicalType().contains("clothing") || m.getLexicalType().contains("colors"))
                            clothes_mult.add(m);
                    }
                }
                int numAgents = agents.size(), numCloth = clothes.size(), numParts = parts.size();


                //Get the number of _attached_ clothes/parts in this caption,
                //as well as adding them to the outtable
                Set<Mention> att_agents = new HashSet<>();
                Set<Mention> att_cloth = new HashSet<>();
                Set<Mention> att_parts = new HashSet<>();
                String capID = d.getID() + "#" + c.getIdx();
                if(attachedMentions.containsKey(capID)){
                    for(Mention[] pair : attachedMentions.get(capID)){
                        att_agents.add(pair[0]);
                        if(pair[1].getLexicalType().equals("bodyparts")){
                            att_parts.add(pair[1]);
                            ot_part.addRow(d.getID(), c.getIdx(), numAgents, numParts,
                                    pair[0].toString(), pair[1].toString(), c.toEntitiesString());
                        } else if(pair[1].getLexicalType().equals("clothing") ||
                                pair[1].getLexicalType().equals("colors") ||
                                pair[1].getLexicalType().equals("clothing/colors")) {
                            att_cloth.add(pair[1]);
                            ot_cloth.addRow(d.getID(), c.getIdx(), numAgents, numCloth,
                                    pair[0].toString(), pair[1].toString(), c.toEntitiesString());
                        }
                    }
                }

                //Add the unattached clothes/parts in this caption to the OutTable
                Set<Mention> unatt_clothes = new HashSet<>(clothes);
                unatt_clothes.removeAll(att_cloth);
                Set<Mention> unatt_parts = new HashSet<>(parts);
                unatt_parts.removeAll(att_parts);
                if(!unatt_clothes.isEmpty())
                    otUnatt_cloth.addRow(d.getID(), c.getIdx(), numAgents, numCloth,
                            StringUtil.listToString(unatt_clothes, "|"), c.toEntitiesString());
                if(!unatt_parts.isEmpty())
                    otUnatt_part.addRow(d.getID(), c.getIdx(), numAgents, numParts,
                            StringUtil.listToString(unatt_parts, "|"), c.toEntitiesString());

                //Increment the appropriate frequencies
                if(numCloth > 0)
                    hist_agentCloth.increment("agents:" + numAgents + ";clothes:" + numCloth);
                if(numParts > 0)
                    hist_agentPart.increment("agents:" + numAgents + ";parts:" + numParts);
                freqs.increment("att_agents", att_agents.size());
                freqs.increment("agents", numAgents);
                freqs.increment("mult_agents", agents_mult.size());
                freqs.increment("att_parts", att_parts.size());
                freqs.increment("parts", numParts);
                freqs.increment("mult_parts", parts_mult.size());
                freqs.increment("att_cloth", att_cloth.size());
                freqs.increment("cloth", numCloth);
                freqs.increment("mult_cloth", clothes_mult.size());

                //Finally, add the multi-type agents/clothes/parts to the appropriate
                //outtables
                if(!agents_mult.isEmpty())
                    otMult_agents.addRow(d.getID(), c.getIdx(),
                            StringUtil.listToString(agents_mult, "|"), c.toEntitiesString());
                if(!parts_mult.isEmpty())
                    otMult_parts.addRow(d.getID(), c.getIdx(),
                            StringUtil.listToString(parts_mult, "|"), c.toEntitiesString());
                if(!clothes_mult.isEmpty())
                    otMult_cloth.addRow(d.getID(), c.getIdx(),
                            StringUtil.listToString(clothes_mult, "|"), c.toEntitiesString());
            }
        }

        Logger.log("Exporting files");
        ot_cloth.writeToCsv("ex_attachedCloth", true);
        ot_part.writeToCsv("ex_attachedParts", true);
        otUnatt_cloth.writeToCsv("ex_unattachedCloth", true);
        otUnatt_part.writeToCsv("ex_unattachedParts", true);
        otMult_agents.writeToCsv("ex_multiTypeAgents", true);
        otMult_cloth.writeToCsv("ex_multiTypeCloth", true);
        otMult_parts.writeToCsv("ex_multiTypeParts", true);
        FileIO.writeFile(hist_agentCloth, "hist_agentCloth", "csv", true);
        FileIO.writeFile(hist_agentPart, "hist_agentPart", "csv", true);

        Logger.log("Mention Statistics");
        List<List<String>> freqTable = new ArrayList<>();
        List<String> columns = new ArrayList<>();
        columns.add(""); columns.add("count"); columns.add("% total"); columns.add("% total + multi");
        freqTable.add(columns);
        String[] total_labels = {"agents", "cloth", "parts"};
        for(String total_label : total_labels){
            String[] labels = {"att_" + total_label, total_label, "mult_" + total_label};
            for(String label : labels){
                List<String> row = new ArrayList<>();
                row.add(label); row.add(String.format("%d", (int)freqs.get(label)));
                row.add(String.format("%.3f%%", 100.0 * freqs.get(label) / freqs.get(total_label)));
                row.add(String.format("%.3f%%", 100.0 * freqs.get(label) / (freqs.get(total_label) + freqs.get("mult_" + total_label))));
                freqTable.add(row);
            }
        }
        System.out.println(StringUtil.toTableStr(freqTable));
    }

    public static void export_clothBodypartStats(Collection<Document> docSet, String dataSplit)
    {
        DoubleDict<String> hist_clothHeads = new DoubleDict<>();
        DoubleDict<String> hist_partHeads = new DoubleDict<>();
        DoubleDict<String> hist_xofy = new DoubleDict<>();

        Map<String, String> xCategories = new HashMap<>();
        String[][] xTable = FileIO.readFile_table(Overlord.dataPath + "xofy_topLemmaX.csv");
        for(String[] row : xTable)
            xCategories.put(row[0], row[2]);


        for(Document d : docSet){
            for(Caption c : d.getCaptionList()){
                List<Mention> mentions = c.getMentionList();
                for(int i=0; i<mentions.size(); i++){
                    Mention m_i = mentions.get(i);
                    String headLem_i = m_i.getHead().getLemma().toLowerCase();

                    //Update the head word frequencies
                    if(m_i.getLexicalType().contains("clothing") || m_i.getLexicalType().contains("colors"))
                        hist_clothHeads.increment(headLem_i);
                    if(m_i.getLexicalType().contains("bodyparts"))
                        hist_partHeads.increment(headLem_i);

                    //determine if there's an XofY construction here
                    if(i < mentions.size() - 1){
                        Mention m_j = mentions.get(i+1);
                        List<Token> interstialToks = c.getInterstitialTokens(m_i, m_j);
                        if(interstialToks.size() == 1 && interstialToks.get(0).toString().equals("of")){
                            //determine what category this X belongs to
                            String xCategory;
                            if(xCategories.containsKey(headLem_i))
                                xCategory = xCategories.get(headLem_i);
                            else if(m_i.getLexicalType().contains("clothing") || m_i.getLexicalType().contains("bodyparts"))
                                xCategory = m_i.getLexicalType();
                            else
                                xCategory = headLem_i;
                            hist_xofy.increment(xCategory);
                        }
                    }
                }
            }


        }

        FileIO.writeFile(hist_clothHeads, "hist_clothHeads_" + dataSplit, "csv", true);
        FileIO.writeFile(hist_partHeads, "hist_partHeads_" + dataSplit, "csv", true);
        FileIO.writeFile(hist_xofy, "hist_xofy_" + dataSplit, "csv", true);
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

    /**Trains an Illinois Chunker and exports a chunker evaluation file
     *
     * @param chunkDir
     * @param trainFile
     * @param evalFile
     * @param outFile
     * @param numIter
     */
    public static void trainAndExportChunkerFiles(String chunkDir, String trainFile,
                                                  String evalFile, String outFile, int numIter)
    {
        Chunker.train(chunkDir + trainFile, chunkDir, numIter);
        Chunker chnkr = new Chunker(chunkDir);
        chnkr.exportToConll(chunkDir + evalFile, chunkDir + outFile);
    }

    /**Given a set of documents, exports part-of-speech and chunk files
     *
     * @param docSet
     * @param posFile
     * @param chunkFile
     */
    public static void exportPosAndChunkData(Collection<Document> docSet, String posFile, String chunkFile)
    {
        List<String> ll_pos = new ArrayList<>();
        for(Document d : docSet)
            for(Caption c : d.getCaptionList())
                ll_pos.add(c.toPosString());
        FileIO.writeFile(ll_pos, posFile, "txt", true);

        List<String> ll_chunk = new ArrayList<>();
        for(Document d : docSet) {
            for (Caption c : d.getCaptionList()) {
                ll_chunk.addAll(c.toConllStrings());
                ll_chunk.add("");
            }
        }
        FileIO.writeFile(ll_chunk, chunkFile, "txt", true);
    }

    /**Trains and tests a part-of-speech tagger
     *
     * @param posDir
     * @param trainFile
     * @param evalFile
     */
    public static void trainAndTestTagger(String posDir, String trainFile, String evalFile)
    {
        Tagger.train(posDir + trainFile, posDir);
        Tagger tggr = new Tagger(posDir);
        tggr.test(posDir + evalFile);
    }

    /**Reads MSCOCO sentences, predicts POS tags / chunk boundaries, and returns
     * caption objects
     *
     * @param data
     * @param posDir
     * @param chunkDir
     */
    public static Map<String, Caption[]> readCocoData(String data, String posDir, String chunkDir)
    {
        Logger.log("Reading COCO data from file");
        Map<String, String[]> captionDict = new HashMap<>();
        List<String> lineList = FileIO.readFile_lineList(data);
        for(String line : lineList){
            String[] lineParts = line.split("\t");
            String[] idParts = lineParts[0].split("#");
            String imgID = idParts[0];
            int capIdx = Integer.parseInt(idParts[1]);

            if(capIdx > 4){
                Logger.log("Found caption with idx >4 " + lineParts[0] + "; skipping");
                continue;
            }

            if(!captionDict.containsKey(imgID))
                captionDict.put(imgID, new String[5]);
            captionDict.get(imgID)[capIdx] = lineParts[1];
        }

        Logger.log("Initializing type lexicons");
        Mention.initLexiconDict(Overlord.lexPath);

        Logger.log("Predicting pos tags and chunks");
        Tagger tggr = new Tagger(posDir);
        Chunker chnkr = new Chunker(chunkDir);
        Map<String, Caption[]> capDict = new HashMap<>();
        for(String docID : captionDict.keySet()){
            Caption[] caps = new Caption[5];
            for(int i=0; i<captionDict.get(docID).length; i++)
                caps[i] = chnkr.predictCaptionChunks(tggr.predict(captionDict.get(docID)[i]), docID, i);
            capDict.put(docID, caps);
        }
        return capDict;
    }
}
