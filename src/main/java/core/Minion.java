package core;

import learn.ClassifyUtil;
import learn.FeatureVector;
import nlptools.IllinoisAnnotator;
import nlptools.WordnetUtil;
import out.OutTable;
import structures.*;
import utilities.*;

import java.awt.geom.Area;
import java.util.*;
import java.util.function.Function;

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

                    Set<String> collectives = new HashSet<>(FileIO.readFile_lineList(Overlord.flickr30kResources + "collectiveNouns.txt"));
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
        System.out.println(StringUtil.toTableStr(freqTable, true));
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
        System.out.println(StringUtil.toTableStr(table, true));
    }


    public static void printGroundingStats(Collection<Document> docSet, String dataset)
    {
        Mention.initializeLexicons(Overlord.flickr30k_lexicon, Overlord.mscoco_lexicon);
        DoubleDict<String> boxStats = new DoubleDict<>();
        Set<String> newCatEntries = new HashSet<>();
        DoubleDict<String> noboxCatDistro = new DoubleDict<>();
        for(Document d : docSet){
            if(dataset.equals("flickr30k")){
                for(Mention m : d.getMentionList()){
                    if(m.getPronounType() != Mention.PRONOUN_TYPE.NONE)
                        continue;

                    if(!m.getChainID().equals("0")){
                        Set<BoundingBox> boxSet = d.getBoxSetForMention(m);
                        if(boxSet.isEmpty())
                            boxStats.increment("vis_nobox");
                        boxStats.increment("vis");
                    } else {
                        boxStats.increment("nonvis");
                    }
                }
            } else if(dataset.equals("mscoco")){
                for(Mention m : d.getMentionList()){
                    if(m.getPronounType() != Mention.PRONOUN_TYPE.NONE)
                        continue;
                    if(!m.getChainID().equals("0")){
                        if(Mention.getLexicalEntry_cocoCategory(m, false) == null){
                            boxStats.increment("vis_nocat");
                            Set<BoundingBox> boxSet = d.getBoxSetForMention(m);
                            for(BoundingBox b : boxSet)
                                newCatEntries.add(b.getCategory() + " -> " + m.getHead().getLemma().toLowerCase());
                        } else {
                            Set<BoundingBox> boxSet = d.getBoxSetForMention(m);
                            if(boxSet.isEmpty()) {
                                boxStats.increment("vis_cat_nobox");
                                for(String catPart : Mention.getLexicalEntry_cocoCategory(m, true).split("/"))
                                    noboxCatDistro.increment(Mention.getSuperCategory(catPart));
                            } else {
                                boxStats.increment("vis_cat_box");
                            }
                            boxStats.increment("vis_cat");
                        }
                    } else {
                        boxStats.increment("nonvis");
                    }
                }
            }
        }

        if(dataset.equals("flickr30k")){
            double totalMentions = boxStats.get("vis") + boxStats.get("nonvis");
            System.out.printf("Visual: %d (%.2f%%); Nonvisual: %d (%.2f%%)\n",
                    (int)boxStats.get("vis"), 100.0 * boxStats.get("vis") / totalMentions,
                    (int)boxStats.get("nonvis"), 100.0 * boxStats.get("nonvis") /
                            totalMentions);
            System.out.printf("Visual Nobox: %d (%.2f%% of vis)\n",
                    (int)boxStats.get("vis_nobox"), 100.0 * boxStats.get("vis_nobox") /
                            boxStats.get("vis"));
        } else if(dataset.equals("mscoco")){
            for(String s : newCatEntries)
                System.out.println(s);

            double totalCatNobox = noboxCatDistro.getSum();
            noboxCatDistro.keySet().forEach(cat -> noboxCatDistro.divide(cat, totalCatNobox));
            Logger.log("Supercategory distro");
            System.out.print(noboxCatDistro);
            double visTotal = boxStats.get("vis_nocat") + boxStats.get("vis_cat");
            double total = visTotal + boxStats.get("nonvis");
            System.out.printf("Visual: %d (%.2f%%); Nonvisual: %d (%.2f%%)\n",
                    (int)visTotal,
                    100.0 * visTotal / total,
                    (int)boxStats.get("nonvis"), 100.0 * boxStats.get("nonvis") /
                            total);
            System.out.printf("Visual Nocat: %d (%.2f%% of vis); "+
                            "Visual Cat Nobox: %d (%.2f%% of vis); "+
                            "Visual Cat Box: %d (%.2f%% of vis)\n",
                    (int)boxStats.get("vis_nocat"), 100.0 * boxStats.get("vis_nocat") /
                            visTotal, (int)boxStats.get("vis_cat_nobox"), 100.0 *
                            boxStats.get("vis_cat_nobox") / visTotal, (int)boxStats.get("vis_cat_box"),
                    100.0 * boxStats.get("vis_cat_box") / visTotal);
        }
    }



    public static void export_bryanPreproc_coco(Collection<Document> docSet)
    {
        String boxFeatFile = Overlord.dataPath + "mscoco/chrisCOCOFeats.csv";
        Logger.log("Loading box features from <%s>", boxFeatFile);
        Set<String> docIDs = new HashSet<>();
        Map<String, FeatureVector> boxFeatures = new HashMap<>();
        for(String[] row : FileIO.readFile_table(boxFeatFile)){
            String docID = row[0].replace("COCO_train2014_", "");
            docIDs.add(docID);
            int boxIdx = Integer.parseInt(row[1]);
            FeatureVector fv = new FeatureVector();
            for(int i=2; i<row.length; i++)
                fv.addFeature(i-1, Double.parseDouble(row[i]));
            boxFeatures.put(docID + ";box:" + boxIdx, fv);
        }

        Logger.log("Retaining only relevant documents");
        Collection<Document> docSubset = new HashSet<>();
        for(Document d : docSet)
            if(docIDs.contains(d.getID()))
                docSubset.add(d);

        Logger.log("Storing in-order lists of box feats, mention strings, and ids");
        List<String> ll_img = new ArrayList<>();
        List<String> ll_txt = new ArrayList<>();
        List<String> ll_ids = new ArrayList<>();
        int docIdx = 0;
        for(Document d : docSubset){
            for(Mention m : d.getMentionList()){
                String normText = m.toString().toLowerCase().trim();
                for(BoundingBox b : d.getBoundingBoxSet()){
                    FeatureVector fv = boxFeatures.get(b.getUniqueID());
                    if(fv == null){
                        Logger.log(new Exception("Found no box features for " + b.getUniqueID()));
                    } else {
                        ll_img.add(StringUtil.listToString(fv.toDenseVector(), ","));
                        ll_txt.add(normText);
                        ll_ids.add(m.getUniqueID() + "|" + b.getUniqueID());
                    }
                }
            }

            docIdx++;
            Logger.logStatus("Completed %.2f%%", 100.0 * docIdx / docSet.size());
        }

        FileIO.writeFile(ll_img, Overlord.dataPath + "feats/img_coco_train_sub", "csv", false);
        FileIO.writeFile(ll_txt, Overlord.dataPath + "feats/txt_coco_train_sub", "txt", false);
        FileIO.writeFile(ll_ids, Overlord.dataPath + "feats/ids_coco_train_sub", "txt", false);
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
        IllinoisAnnotator.trainChunker(chunkDir + trainFile, numIter, chunkDir);
        IllinoisAnnotator annotator = IllinoisAnnotator.createChunker(chunkDir);
        annotator.testChunker_exportToConll(chunkDir + evalFile, chunkDir + outFile);
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
        IllinoisAnnotator.trainTagger(posDir + trainFile, posDir);
        IllinoisAnnotator annotator = IllinoisAnnotator.createTagger(posDir);
        annotator.testTagger(posDir + evalFile);
    }

    public static void export_cocoCategoryStats_givenBox(Collection<Document> docSet)
    {
        Mention.initializeLexicons(Overlord.flickr30k_lexicon, Overlord.mscocoResources + "coco_lex.csv");
        Set<String> supercategories = Mention.getCOCOSupercategories();
        Set<String> categories = Mention.getCOCOCategories();

        //Create a mapping of categories to get the frequency of possible mentions
        Map<String, Set<Caption>> zeroMentionCaps = new HashMap<>();
        Map<String, DoubleDict<Integer>> catPossMentions_strict_perDoc = new HashMap<>();
        Map<String, DoubleDict<Integer>> catPossMentions_all_perDoc = new HashMap<>();
        Map<String, DoubleDict<Integer>> catPossMentions_strict_perCap = new HashMap<>();
        Map<String, DoubleDict<Integer>> catPossMentions_all_perCap = new HashMap<>();
        for(String cat : categories){
            catPossMentions_strict_perDoc.put(cat, new DoubleDict<>());
            catPossMentions_all_perDoc.put(cat, new DoubleDict<>());
            catPossMentions_strict_perCap.put(cat, new DoubleDict<>());
            catPossMentions_all_perCap.put(cat, new DoubleDict<>());
            zeroMentionCaps.put(cat, new HashSet<>());
        }

        Logger.log("Computing box-mention-category statistics");
        for(Document d : docSet){
            //Set up the possible mentions for each category (per cap and per doc)
            Map<String, Set<Mention>> catMentionDict_strict_perDoc = new HashMap<>();
            Map<String, Set<Mention>> catMentionDict_all_perDoc = new HashMap<>();
            Map<String, Set<Mention>[]> catMentionDict_strict_perCap = new HashMap<>();
            Map<String, Set<Mention>[]> catMentionDict_all_perCap = new HashMap<>();
            for(String cat : categories){
                catMentionDict_strict_perDoc.put(cat, new HashSet<>());
                catMentionDict_all_perDoc.put(cat, new HashSet<>());
                catMentionDict_strict_perCap.put(cat, new HashSet[5]);
                catMentionDict_all_perCap.put(cat, new HashSet[5]);
                for(int i=0; i<5; i++) {
                    catMentionDict_strict_perCap.get(cat)[i] = new HashSet<>();
                    catMentionDict_all_perCap.get(cat)[i] = new HashSet<>();
                }
            }

            //Partition the mentions into category groupings
            for(Mention m : d.getMentionList()){
                //for implementation ease (since these will not
                //change our statistical analysis much) skip those
                //rare sixth captions
                if(m.getCaptionIdx() > 4)
                    continue;

                String category_strict = Mention.getLexicalEntry_cocoCategory(m, false);
                String category_all = Mention.getLexicalEntry_cocoCategory(m, true);
                if(category_strict != null){
                    for(String cat : category_strict.split("/")){
                        catMentionDict_strict_perDoc.get(cat).add(m);
                        catMentionDict_strict_perCap.get(cat)[m.getCaptionIdx()].add(m);
                    }
                }
                if(category_all != null){
                    for(String cat : category_all.split("/")){
                        catMentionDict_all_perDoc.get(cat).add(m);
                        catMentionDict_all_perCap.get(cat)[m.getCaptionIdx()].add(m);
                    }
                }
            }

            //Iterate through each bounding box, incrementing the possible number of mentions
            //for this category (by caption and by document)
            for(BoundingBox b : d.getBoundingBoxSet()){
                String cat = b.getCategory();
                catPossMentions_strict_perDoc.get(cat).increment(catMentionDict_strict_perDoc.get(cat).size());
                catPossMentions_all_perDoc.get(cat).increment(catMentionDict_all_perDoc.get(cat).size());
                for(int i=0; i<5; i++){
                    catPossMentions_strict_perCap.get(cat).increment(catMentionDict_strict_perCap.get(cat)[i].size());
                    catPossMentions_all_perCap.get(cat).increment(catMentionDict_all_perCap.get(cat)[i].size());

                    if(catMentionDict_strict_perCap.get(cat)[i].size() == 0)
                        zeroMentionCaps.get(cat).add(d.getCaption(i));
                }
            }
        }

        Logger.log("Exporting statistics and charts to file");
        OutTable ot_avg = new OutTable("category", "avg poss mentions (strict; per cap)",
                "avg poss mentions (all; per cap)", "avg poss mentions (strict; per doc)",
                "avg poss mentions (all; per doc)", "total boxes",
                "% w/ 0 poss mentions (strict, per cap)", "% w/ 0 poss mentions (all; per cap)",
                "% w/ 0 poss mentions (strict; per doc)", "% w/ 0 poss mentions (all; per doc)");
        OutTable ot_catHist = new OutTable("category", "poss mentions", "frequency (strict; per cap)",
                "frequency (all; per cap)", "frequency (strict; per doc)", "frequency (all; per doc)");
        for(String cat : categories){
            Set<Integer> possMentions = new HashSet<>(catPossMentions_all_perCap.get(cat).keySet());
            possMentions.addAll(catPossMentions_all_perDoc.get(cat).keySet());
            possMentions.addAll(catPossMentions_strict_perCap.get(cat).keySet());
            possMentions.addAll(catPossMentions_strict_perDoc.get(cat).keySet());

            for(Integer pm : possMentions)
                ot_catHist.addRow(cat, pm, catPossMentions_strict_perCap.get(cat).get(pm),
                                  catPossMentions_all_perCap.get(cat).get(pm),
                                  catPossMentions_strict_perDoc.get(cat).get(pm),
                                  catPossMentions_all_perDoc.get(cat).get(pm));

            Map<Comparable[], Double> chartData = new HashMap<>();
            for(Integer pm : possMentions) {
                chartData.put(new Comparable[]{"strict_perCap", pm},
                        catPossMentions_strict_perCap.get(cat).get(pm));
                chartData.put(new Comparable[]{"all_perCap", pm},
                        catPossMentions_all_perCap.get(cat).get(pm));
                chartData.put(new Comparable[]{"strict_perDoc", pm},
                        catPossMentions_strict_perDoc.get(cat).get(pm));
                chartData.put(new Comparable[]{"all_perDoc", pm},
                        catPossMentions_all_perDoc.get(cat).get(pm));
            }
            ChartIO.saveBarChart(chartData, "out/charts/" + cat.replace(" ", "") +
                    "_" + Util.getCurrentDateTime("yyyyMMdd") + ".jpg");

            double weightedMean_strict_perCap = 0.0, weightedMean_all_perCap = 0.0;
            double weightedMean_strict_perDoc = 0.0, weightedMean_all_perDoc = 0.0;
            //Annotations should be the same regardless of strict / all
            double totalAnnotations_perDoc = catPossMentions_strict_perDoc.get(cat).getSum();
            double totalAnnotations_perCap = catPossMentions_strict_perCap.get(cat).getSum();

            for(Integer i : possMentions){
                weightedMean_strict_perCap += (i+1) * catPossMentions_strict_perCap.get(cat).get(i);
                weightedMean_all_perCap += (i+1) * catPossMentions_all_perCap.get(cat).get(i);
                weightedMean_strict_perDoc += (i+1) * catPossMentions_strict_perDoc.get(cat).get(i);
                weightedMean_all_perDoc += (i+1) * catPossMentions_all_perDoc.get(cat).get(i);
            }
            weightedMean_strict_perCap /= totalAnnotations_perCap;
            weightedMean_all_perCap /= totalAnnotations_perCap;
            weightedMean_strict_perDoc /= totalAnnotations_perDoc;
            weightedMean_all_perDoc /= totalAnnotations_perDoc;
            weightedMean_strict_perCap--;
            weightedMean_strict_perDoc--;
            weightedMean_all_perCap--;
            weightedMean_all_perDoc--;

            ot_avg.addRow(cat, weightedMean_strict_perCap, weightedMean_all_perCap,
                    weightedMean_strict_perDoc, weightedMean_all_perDoc, totalAnnotations_perDoc,
                    catPossMentions_strict_perCap.get(cat).get(0) / totalAnnotations_perCap,
                    catPossMentions_all_perCap.get(cat).get(0) / totalAnnotations_perCap,
                    catPossMentions_strict_perDoc.get(cat).get(0) / totalAnnotations_perDoc,
                    catPossMentions_all_perDoc.get(cat).get(0) / totalAnnotations_perDoc);
        }
        ot_avg.writeToCsv("hist_cocoCategoryStats_givenBox", true);
        ot_catHist.writeToCsv("hist_cocoCategoryHist_givenBox", true);


        //Print supercategory statistics
        List<List<String>> table = new ArrayList<>();
        table.add(Arrays.asList("super_cat", "strict; perCap", "strict; perDoc", "all; perCap", "all: perDoc"));
        Map<String, Set<String>> supercategoryDict = new HashMap<>();
        for(String category : categories){
            String supercategory = Mention.getSuperCategory(category);
            if(!supercategoryDict.containsKey(supercategory))
                supercategoryDict.put(supercategory, new HashSet<>());
            supercategoryDict.get(supercategory).add(category);
        }
        for(String superCat : supercategories){
            double wMean_strict_perCap = 0.0, wMean_all_perCap = 0.0;
            double wMean_strict_perDoc = 0.0, wMean_all_perDoc = 0.0;
            double totalPerCap = 0.0, totalPerDoc = 0.0;
            for(String cat : supercategoryDict.get(superCat)){
                Set<Integer> possMentions = new HashSet<>(catPossMentions_all_perCap.get(cat).keySet());
                possMentions.addAll(catPossMentions_all_perDoc.get(cat).keySet());
                possMentions.addAll(catPossMentions_strict_perCap.get(cat).keySet());
                possMentions.addAll(catPossMentions_strict_perDoc.get(cat).keySet());

                for(Integer i : possMentions){
                    wMean_strict_perCap += (i+1) * catPossMentions_strict_perCap.get(cat).get(i);
                    wMean_all_perCap += (i+1) * catPossMentions_all_perCap.get(cat).get(i);
                    wMean_strict_perDoc += (i+1) * catPossMentions_strict_perDoc.get(cat).get(i);
                    wMean_all_perDoc += (i+1) * catPossMentions_all_perDoc.get(cat).get(i);
                }
                totalPerCap += catPossMentions_strict_perCap.get(cat).getSum();
                totalPerDoc += catPossMentions_strict_perDoc.get(cat).getSum();
            }

            table.add(Arrays.asList(superCat, String.format("%.2f", wMean_strict_perCap / totalPerCap - 1),
                    String.format("%.2f", wMean_strict_perDoc / totalPerDoc - 1),
                    String.format("%.2f", wMean_all_perCap / totalPerCap - 1),
                    String.format("%.2f",wMean_all_perDoc / totalPerDoc - 1)));
        }
        System.out.println(StringUtil.toTableStr(table, true));
    }

    public static void export_cocoCategoryStats_givenMention(Collection<Document> docSet)
    {
        Mention.initializeLexicons(Overlord.flickr30k_lexicon, Overlord.mscocoResources + "coco_lex.csv");
        Set<String> supercategories = Mention.getCOCOSupercategories();
        Set<String> categories = Mention.getCOCOCategories();

        //Create a mapping of categories to get the frequency of possible mentions
        Map<String, DoubleDict<Integer>> catPossBoxes_strict = new HashMap<>();
        Map<String, DoubleDict<Integer>> catPossBoxes_all = new HashMap<>();
        for(String cat : categories){
            catPossBoxes_strict.put(cat, new DoubleDict<>());
            catPossBoxes_all.put(cat, new DoubleDict<>());
        }

        Logger.log("Computing box-mention-category statistics");
        for(Document d : docSet){
            //Get a mapping of each category and box
            Map<String, Set<BoundingBox>> catBoxDict = new HashMap<>();
            for(String cat : categories)
                catBoxDict.put(cat, new HashSet<>());
            for(BoundingBox b : d.getBoundingBoxSet())
                catBoxDict.get(b.getCategory()).add(b);

            //For mention, determine its strict and all type
            //and increment histograms appropriately
            for(Mention m : d.getMentionList()){
                String category_strict = Mention.getLexicalEntry_cocoCategory(m, false);
                String category_all = Mention.getLexicalEntry_cocoCategory(m, true);
                if(category_strict != null){
                    for(String cat : category_strict.split("/")){
                        catPossBoxes_strict.get(cat).increment(catBoxDict.get(cat).size());
                    }
                }
                if(category_all != null){
                    for(String cat : category_all.split("/")){
                        catPossBoxes_all.get(cat).increment(catBoxDict.get(cat).size());
                    }
                }
            }
        }

        Logger.log("Exporting statistics and charts to file");
        OutTable ot_avg = new OutTable("category", "avg poss boxes (strict)",
                "avg poss boxes (all)", "total mentions",
                "% w/ 0 poss boxes (strict)", "% w/ 0 poss boxes (all)");
        OutTable ot_catHist = new OutTable("category", "poss boxes", "frequency (strict)", "frequency (all)");
        for(String cat : categories){
            Set<Integer> possBoxes = new HashSet<>(catPossBoxes_all.get(cat).keySet());
            possBoxes.addAll(catPossBoxes_strict.get(cat).keySet());

            for(Integer pb : possBoxes)
                ot_catHist.addRow(cat, pb, catPossBoxes_strict.get(cat).get(pb),
                        catPossBoxes_all.get(cat).get(pb));

            Map<Comparable[], Double> chartData = new HashMap<>();
            for(Integer pb : possBoxes) {
                chartData.put(new Comparable[]{"strict", pb},
                        catPossBoxes_strict.get(cat).get(pb));
                chartData.put(new Comparable[]{"all", pb},
                        catPossBoxes_all.get(cat).get(pb));
            }
            ChartIO.saveBarChart(chartData, "out/charts/" + cat.replace(" ", "") +
                    "_" + Util.getCurrentDateTime("yyyyMMdd") + ".jpg");

            double wMean_strict = 0.0, wMean_all = 0.0;

            //Annotations should be the same regardless of strict / all
            double numMentions = catPossBoxes_strict.get(cat).getSum();

            for(Integer i : possBoxes){
                wMean_strict += (i+1) * catPossBoxes_strict.get(cat).get(i);
                wMean_all += (i+1) * catPossBoxes_all.get(cat).get(i);
            }
            wMean_strict /= numMentions;
            wMean_all /= numMentions;
            wMean_strict--;
            wMean_all--;

            ot_avg.addRow(cat, wMean_strict , wMean_all, numMentions,
                    catPossBoxes_strict.get(cat).get(0) / numMentions,
                    catPossBoxes_all.get(cat).get(0) / numMentions);
        }
        ot_avg.writeToCsv("hist_cocoCategoryStats_givenMention", true);
        ot_catHist.writeToCsv("hist_cocoCategoryHist_givenMention", true);


        //Print supercategory statistics
        List<List<String>> table = new ArrayList<>();
        table.add(Arrays.asList("super_cat", "strict", "all"));
        Map<String, Set<String>> supercategoryDict = new HashMap<>();
        for(String category : categories){
            String supercategory = Mention.getSuperCategory(category);
            if(!supercategoryDict.containsKey(supercategory))
                supercategoryDict.put(supercategory, new HashSet<>());
            supercategoryDict.get(supercategory).add(category);
        }
        for(String superCat : supercategories){
            double wMean_strict = 0.0, wMean_all = 0.0, numMentions = 0.0;
            for(String cat : supercategoryDict.get(superCat)){
                Set<Integer> possBoxes = new HashSet<>(catPossBoxes_strict.get(cat).keySet());
                possBoxes.addAll(catPossBoxes_all.get(cat).keySet());

                for(Integer i : possBoxes){
                    wMean_strict += (i+1) * catPossBoxes_strict.get(cat).get(i);
                    wMean_all += (i+1) * catPossBoxes_all.get(cat).get(i);
                }
                numMentions += catPossBoxes_strict.get(cat).getSum();
            }

            table.add(Arrays.asList(superCat, String.format("%.2f", wMean_strict / numMentions - 1),
                    String.format("%.2f", wMean_all / numMentions - 1)));
        }
        System.out.println(StringUtil.toTableStr(table, true));
    }

    /**Computes category stats to determine to what extent our MSCOCO lexicons
     * cover the data appropriately
     *
     * @param docSet
     */
    public static void export_cocoCategoryStats_coverage(Collection<Document> docSet)
    {
        Mention.initializeLexicons(Overlord.flickr30k_lexicon, Overlord.mscocoResources + "coco_lex.csv");
        Set<String> categories = Mention.getCOCOCategories();

        Set<String> nonvisHeads = new HashSet<>();
        for(String[] row : FileIO.readFile_table(Overlord.flickr30kResources + "hist_nonvisual.csv")){
            if(Double.parseDouble(row[1]) > 10){
                String nonvisHead = row[0].toLowerCase().trim();
                //We don't want to include any heads that have
                //a coco category (eg. 'control')
                if(Mention.getLexicalEntry_cocoCategory(nonvisHead, true) == null)
                    nonvisHeads.add(row[0].toLowerCase().trim());
            }
        }

        DoubleDict<String> hist = new DoubleDict<>();
        Map<String, List<Double>> mentionCounts = new HashMap<>();
        String[] mentionKeys = {"perCap", "nonvisPerCap", "nonvisPerDoc",
                "coveredPerCap", "uncoveredPerCap",
                "scenePerCap", "clothColorsPerCap",
                "clothColorsPerDoc", "bodypartsPerCap",
                "bodypartsPerDoc", "instrumentsPerCap",
                "instrumentsPerDoc", "coveredPerDoc",
                "uncoveredPerDoc", "scenePerDoc",
                "perDoc", "pronomPerCap", "pronomPerDoc"};
        for(String key : mentionKeys)
            mentionCounts.put(key, new ArrayList<>());

        DoubleDict<String> uncoveredHeads = new DoubleDict<>();

        DoubleDict<String> imgs_head = new DoubleDict<>();
        DoubleDict<String> imgs_cat = new DoubleDict<>();
        DoubleDict<String> imgs_joint = new DoubleDict<>();
        for(Document d : docSet){
            Set<String> headsPresent = new HashSet();
            Set<String> catsPresent = new HashSet<>();
            for(BoundingBox b : d.getBoundingBoxSet())
                catsPresent.add(b.getCategory());

            mentionCounts.get("perDoc").add((double)d.getMentionList().size());
            int covered_perDoc = 0, unc_perDoc = 0, scene_perDoc = 0, pronom_perDoc = 0;
            int clothColor_perDoc = 0, part_perDoc = 0, instr_perDoc = 0;
            int nonvisPerDoc = 0;

            for(Caption c : d.getCaptionList()){
                mentionCounts.get("perCap").add((double)c.getMentionList().size());
                int covered_perCap = 0, unc_perCap = 0, scene_perCap = 0, pronom_perCap = 0;
                int clothColor_perCap = 0, part_perCap = 0, instr_perCap = 0;
                int nonvisPerCap = 0;

                for(Mention m : c.getMentionList()) {
                    hist.increment("mention_count");
                    String head = m.getHead().getLemma().toLowerCase();

                    String category = Mention.getLexicalEntry_cocoCategory(m, true);
                    String type = Mention.getLexicalEntry_flickr(m);

                    if(m.getPronounType() != Mention.PRONOUN_TYPE.NONE){
                        hist.increment("pronom");
                        pronom_perCap++; pronom_perDoc++;
                    } else if(nonvisHeads.contains(head)){
                        hist.increment("nonvisual");
                        nonvisPerCap++; nonvisPerDoc++;
                    } else if (type != null && type.equals("scene")) {
                        hist.increment("scene");
                        scene_perCap++; scene_perDoc++;
                    } else if(type != null && (type.equals("clothing") ||
                            type.equals("colors") || type.equals("clothing/colors")) &&
                            (category == null || !category.contains("tie"))) {
                        hist.increment("clothColors");
                        clothColor_perCap++; clothColor_perDoc++;
                    } else if(type != null && type.equals("bodyparts")){
                        hist.increment("bodyparts");
                        part_perCap++; part_perDoc++;
                    } else if(type != null && type.equals("instruments")){
                        hist.increment("instruments");
                        instr_perCap++; instr_perDoc++;
                    } else if (categories != null) {
                        hist.increment("covered");
                        covered_perCap++;
                        covered_perDoc++;
                    } else {
                        hist.increment("uncovered");
                        unc_perCap++; unc_perDoc++;
                        uncoveredHeads.increment(head);
                        headsPresent.add(head);
                    }
                }
                mentionCounts.get("coveredPerCap").add((double)covered_perCap);
                mentionCounts.get("uncoveredPerCap").add((double)unc_perCap);
                mentionCounts.get("scenePerCap").add((double)scene_perCap);
                mentionCounts.get("pronomPerCap").add((double)pronom_perCap);
                mentionCounts.get("clothColorsPerCap").add((double)clothColor_perCap);
                mentionCounts.get("bodypartsPerCap").add((double)part_perCap);
                mentionCounts.get("instrumentsPerCap").add((double)instr_perCap);
                mentionCounts.get("nonvisPerCap").add((double)nonvisPerCap);
            }
            mentionCounts.get("coveredPerDoc").add((double)covered_perDoc);
            mentionCounts.get("uncoveredPerDoc").add((double)unc_perDoc);
            mentionCounts.get("scenePerDoc").add((double)scene_perDoc);
            mentionCounts.get("pronomPerDoc").add((double)pronom_perDoc);
            mentionCounts.get("clothColorsPerDoc").add((double)clothColor_perDoc);
            mentionCounts.get("bodypartsPerDoc").add((double)part_perDoc);
            mentionCounts.get("instrumentsPerDoc").add((double)instr_perDoc);
            mentionCounts.get("nonvisPerDoc").add((double)nonvisPerDoc);

            for(String head : headsPresent)
                imgs_head.increment(head);
            for(String cat : catsPresent)
                imgs_cat.increment(cat.replace(" ", "_").toUpperCase());
            for(String head : headsPresent)
                for(String cat : catsPresent)
                    imgs_joint.increment(cat.replace(" ", "_").toUpperCase() + "|" + head);
        }

        //remove everything lower than the cutoff
        int frequencyCuttoff = 5;
        Set<String> headsToRemove = new HashSet<>();
        for(String head : imgs_head.keySet())
            if(imgs_head.get(head) < frequencyCuttoff)
                headsToRemove.add(head);
        for(String head : headsToRemove)
            imgs_head.remove(head);

        for(String key : imgs_head.keySet())
            imgs_head.divide(key, docSet.size());
        for(String key : imgs_cat.keySet())
            imgs_cat.divide(key, docSet.size());
        for(String key : imgs_joint.keySet())
            imgs_joint.divide(key, docSet.size());

        OutTable ot_pmi = new OutTable("category", "head", "cat_prob", "head_prob", "joint_prob", "pmi");
        for(String cat : imgs_cat.keySet()){
            double catProb = imgs_cat.get(cat);
            for(String head : imgs_head.keySet()){
                double headProb = imgs_head.get(head);
                double jointProb = imgs_joint.get(cat + "|" + head);
                if(jointProb > 0){
                    double pmi = StatisticalUtil.computePMI(catProb, headProb, jointProb, true);
                    ot_pmi.addRow(cat, head, catProb, headProb, jointProb, pmi);
                }
            }
        }
        ot_pmi.writeToCsv("hist_catHeadPmi", true);

        System.out.printf("Mention distribution - pronominal: %.2f%%; nonvis: %.2f%%; "+
                        "scene: %.2f%%; cloth/colors: %.2f%%; bodyparts: %.2f%%; "+
                        "instruments: %.2f%%; covered: %.2f%%; uncovered: %.2f%%\n",
                100.0 * hist.get("pronom") / hist.get("mention_count"),
                100.0 * hist.get("nonvisual") / hist.get("mention_count"),
                100.0 * hist.get("scene") / hist.get("mention_count"),
                100.0 * hist.get("clothColors") / hist.get("mention_count"),
                100.0 * hist.get("bodyparts") / hist.get("mention_count"),
                100.0 * hist.get("instruments") / hist.get("mention_count"),
                100.0 * hist.get("covered") / hist.get("mention_count"),
                100.0 * hist.get("uncovered") / hist.get("mention_count"));
        System.out.printf("Mentions per caption: %.2f; pronom: %.2f; nonvis: %.2f; "+
                        "scene: %.2f; cloth/colors: %.2f; bodyparts: %.2f; "+
                        "instruments: %.2f; covered: %.2f; uncovered: %.2f\n",
                StatisticalUtil.getMean(mentionCounts.get("perCap")),
                StatisticalUtil.getMean(mentionCounts.get("pronomPerCap")),
                StatisticalUtil.getMean(mentionCounts.get("nonvisPerCap")),
                StatisticalUtil.getMean(mentionCounts.get("scenePerCap")),
                StatisticalUtil.getMean(mentionCounts.get("clothColorsPerCap")),
                StatisticalUtil.getMean(mentionCounts.get("bodypartsPerCap")),
                StatisticalUtil.getMean(mentionCounts.get("instrumentsPerCap")),
                StatisticalUtil.getMean(mentionCounts.get("coveredPerCap")),
                StatisticalUtil.getMean(mentionCounts.get("uncoveredPerCap")));
        System.out.printf("Mentions per doc: %.2f; pronom: %.2f; nonvis: %.2f; "+
                        "scene: %.2f; cloth/colors: %.2f; bodyparts: %.2f; "+
                        "instruments: %.2f; covered: %.2f; uncovered: %.2f\n",
                StatisticalUtil.getMean(mentionCounts.get("perDoc")),
                StatisticalUtil.getMean(mentionCounts.get("pronomPerDoc")),
                StatisticalUtil.getMean(mentionCounts.get("nonvisPerDoc")),
                StatisticalUtil.getMean(mentionCounts.get("scenePerDoc")),
                StatisticalUtil.getMean(mentionCounts.get("clothColorsPerDoc")),
                StatisticalUtil.getMean(mentionCounts.get("bodypartsPerDoc")),
                StatisticalUtil.getMean(mentionCounts.get("instrumentsPerDoc")),
                StatisticalUtil.getMean(mentionCounts.get("coveredPerDoc")),
                StatisticalUtil.getMean(mentionCounts.get("uncoveredPerDoc")));

        Set<String> headsToDelete = new HashSet<>();
        for(String head : uncoveredHeads.keySet())
            if(uncoveredHeads.get(head) < 6)
                headsToDelete.add(head);
        for(String head : headsToDelete)
            uncoveredHeads.remove(head);
        OutTable ot_uncHeads = new OutTable("head", "freq");
        for(String head : uncoveredHeads.getSortedByValKeys(true))
            ot_uncHeads.addRow(head, uncoveredHeads.get(head));
        ot_uncHeads.writeToCsv("hist_uncoveredCocoHeads", true);
    }


    /**
     *
     * @param docSet
     * @param maxIter
     */
    public static void runCocoCatDistroSim(Collection<Document> docSet, int maxIter, boolean withReplacement)
    {
        //First, get the true category and supercategory distribution
        DoubleDict<String> categoryDistro = new DoubleDict<>();
        DoubleDict<String> supercategoryDistro = new DoubleDict<>();
        for(Document d : docSet){
            for(BoundingBox b : d.getBoundingBoxSet()){
                categoryDistro.increment(b.getCategory());
                supercategoryDistro.increment(b.getSuperCategory());
            }
        }
        double totalBoxes = categoryDistro.getSum();
        for(String cat : categoryDistro.keySet())
            categoryDistro.divide(cat, totalBoxes);
        for(String supercat : supercategoryDistro.keySet())
            supercategoryDistro.divide(supercat, totalBoxes);

        //Sort the categories and supercategories for
        //display consistency
        List<String> categoryList =
                new ArrayList<>(categoryDistro.keySet());
        List<String> supercategoryList =
                new ArrayList<>(supercategoryDistro.keySet());
        Collections.sort(categoryList);
        Collections.sort(supercategoryList);

        Logger.log("True Supercategory Distribution");
        List<String> supercategoryStrList = new ArrayList<>();
        for(String supercategory : supercategoryList){
            supercategoryStrList.add(supercategory);
            supercategoryStrList.add(String.format("%.2f%%",
                    100.0 * supercategoryDistro.get(supercategory)));
        }
        System.out.println(StringUtil.toTableStr(
                Util.listToRows(supercategoryStrList, 8), false));

        Logger.log("True Category Distribution");
        List<String> categoryStrList = new ArrayList<>();
        for(String category : categoryList){
            categoryStrList.add(category);
            categoryStrList.add(String.format("%.2f%%",
                    100.0 * categoryDistro.get(category)));
        }
        System.out.println(StringUtil.toTableStr(
                Util.listToRows(categoryStrList, 8), false));


        DoubleDict<String> categoryHist_sample = new DoubleDict<>();
        DoubleDict<String> supercategoryHist_sample = new DoubleDict<>();
        Logger.log("Randomly sampling for " + maxIter + " iterations");
        List<Document> docList = new ArrayList<>(docSet);
        Random r = new Random();
        List<Double> categoryDivergence = new ArrayList<>();
        List<Double> supercategoryDivergence = new ArrayList<>();
        for(int iter = 0; iter < maxIter; iter++) {
            Document d = docList.get(r.nextInt(docList.size()));
            if(!withReplacement)
                docList.remove(d);

            for(BoundingBox b : d.getBoundingBoxSet()){
                categoryHist_sample.increment(b.getCategory());
                supercategoryHist_sample.increment(b.getSuperCategory());
            }

            //Every hundred images, check the distributions
            if((iter+1) % 100 == 0){
                Logger.log("Iteration " + (iter+1));
                double totalBoxes_sample = categoryHist_sample.getSum();
                DoubleDict<String> categoryDistro_sample = new DoubleDict<>();
                for(String cat : categoryHist_sample.keySet())
                    categoryDistro_sample.increment(cat,
                            categoryHist_sample.get(cat) / totalBoxes_sample);
                DoubleDict<String> supercategoryDistro_sample = new DoubleDict<>();
                for(String supercat : supercategoryHist_sample.keySet())
                    supercategoryDistro_sample.increment(supercat,
                            supercategoryHist_sample.get(supercat) / totalBoxes_sample);

                Logger.log("Sampled Supercategory Distribution");
                double supercatDiv =
                        StatisticalUtil.computeKLDivergence(supercategoryDistro,
                                supercategoryDistro_sample);
                System.out.printf("KL Divergence from truth: %.3f\n", supercatDiv);
                supercategoryDivergence.add(supercatDiv);
                List<String> supercategoryStrList_sample = new ArrayList<>();
                for(String supercategory : supercategoryList){
                    supercategoryStrList_sample.add(supercategory);
                    supercategoryStrList_sample.add(String.format("%.2f%%",
                            100.0 * supercategoryDistro_sample.get(supercategory)));
                }
                System.out.println(StringUtil.toTableStr(
                        Util.listToRows(supercategoryStrList_sample, 8), false));

                Logger.log("Sampled Category Distribution");
                double catDiv =
                        StatisticalUtil.computeKLDivergence(categoryDistro,
                                categoryDistro_sample);
                System.out.printf("KL Divergence from truth: %.3f\n", catDiv);
                categoryDivergence.add(catDiv);
                List<String> categoryStrList_sample = new ArrayList<>();
                for(String category : categoryList){
                    categoryStrList_sample.add(category);
                    categoryStrList_sample.add(String.format("%.2f%%",
                            100.0 * categoryDistro_sample.get(category)));
                }
                System.out.println(StringUtil.toTableStr(
                        Util.listToRows(categoryStrList_sample, 8), false));
            }
        }
        OutTable ot_div = new OutTable("iteration", "supercat_divergence", "category_divergence");
        for(int i=0; i<categoryDivergence.size(); i++)
            ot_div.addRow(i*100, supercategoryDivergence.get(i), categoryDivergence.get(i));
        ot_div.writeToCsv("coco_category_divergence", true);
    }
}
