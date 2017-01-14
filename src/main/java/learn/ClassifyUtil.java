package learn;

import core.Minion;
import core.Overlord;
import nlptools.Word2VecUtil;
import nlptools.WordnetUtil;
import statistical.ScoreDict;
import structures.*;
import utilities.*;

import java.io.*;
import java.util.*;

public abstract class ClassifyUtil {
    private static final int UNK = 0;
    private static final int FALSE = 0;
    private static final int TRUE = 1;

    protected static String _outroot;
    protected static final String PTRN_APPOS = "^NP , (NP (VP |ADJP |PP |and )*)+,.*$";
    protected static final String PTRN_LIST = "^NP , (NP ,?)* and NP.*$";

    /* Static collections used by subordinate threads */
    protected static Map<String, DoubleDict<String>> _imgLemmaCountDict;
    protected static Map<Mention, Chunk[]> _mentionChunkNeighborDict;
    protected static Map<String, Set<String>> _hypDict;
    protected static Set<Mention> _onlyTypeMentions;
    protected static Map<Mention, Chunk> _subjOfDict;
    protected static Map<Mention, Chunk> _objOfDict;
    protected static Word2VecUtil _w2vUtil;
    protected static Map<Mention, Chain> _mentionChainDict;
    protected static Set<String> _mentionPairsWithSubsetBoxes;
    private static Map<String, String> _clothAttrLex;
    protected static Map<Mention, String> _prepDict_left;
    protected static Map<Mention, String> _prepDict_right;

    //lists from files
    private static Set<String> _colors;
    protected static Set<String> _detSet_singular;
    protected static Set<String> _detSet_plural;
    protected static List<String> _detList;
    protected static Set<String> _stopWords;
    protected static List<String> _hypernyms;

    //onehot lists
    protected static List<String> _typePairList;
    protected static List<String> _leftPairList;
    protected static List<String> _rightPairList;
    protected static List<String> _headPairList;
    protected static List<String> _lemmaPairList;
    protected static List<String> _subjOfPairList;
    protected static List<String> _objOfPairList;
    protected static List<String> _headList;
    protected static List<String> _typeList;
    protected static List<String> _leftList;
    protected static List<String> _rightList;
    protected static List<String> _subjOfList;
    protected static List<String> _objOfList;
    protected static List<String> _cardList;
    protected static List<String> _pronounList;
    protected static List<String> _pronounTypeList;
    protected static List<String> _modifierPairs;
    protected static List<String> _modifierList;
    protected static List<String> _numericList;
    protected static List<String> _numericPairs;
    protected static List<String> _prepositionList;
    protected static List<String> _prepositionPairList;

    protected static final String pattern_aside = "^NP , (NP (VP |ADJP |PP |and )*)+,.*$";

    public static void initLists() {
        //onehot lists
        _typePairList = getOneHotList(Overlord.resourcesDir + "hist_typePair_ordered.csv", 1000);
        _leftPairList = getOneHotList(Overlord.resourcesDir + "hist_leftPair_ordered.csv", 1000);
        _rightPairList = getOneHotList(Overlord.resourcesDir + "hist_rightPair_ordered.csv", 1000);
        _headPairList = getOneHotList(Overlord.resourcesDir + "hist_headPair_ordered.csv", 1);
        _lemmaPairList = getOneHotList(Overlord.resourcesDir + "hist_lemmaPair_ordered.csv", 1);
        _subjOfPairList = getOneHotList(Overlord.resourcesDir + "hist_subjOfPair_ordered.csv", 1);
        _objOfPairList = getOneHotList(Overlord.resourcesDir + "hist_objOfPair_ordered.csv", 1);
        _modifierPairs = getOneHotList(Overlord.resourcesDir + "hist_modifierPair.csv", 1);
        _numericPairs = getOneHotList(Overlord.resourcesDir + "hist_numericModifierPair.csv", 1);
        _prepositionPairList = getOneHotList(Overlord.resourcesDir + "hist_prepositionPair.csv", 1);
        _headList = getOneHotList(Overlord.resourcesDir + "hist_head.csv", 1);
        _modifierList = getOneHotList(Overlord.resourcesDir + "hist_modifier.csv", 1);
        _numericList = getOneHotList(Overlord.resourcesDir + "hist_numericModifier.csv", 1);
        _prepositionList = getOneHotList(Overlord.resourcesDir + "hist_preposition.csv", 1);
        _typeList = getOneHotList(Overlord.resourcesDir + "hist_type.csv", 1000);
        _leftList = getOneHotList(Overlord.resourcesDir + "hist_left.csv", 1000);
        _rightList = getOneHotList(Overlord.resourcesDir + "hist_right.csv", 1000);
        _subjOfList = getOneHotList(Overlord.resourcesDir + "hist_subjOf.csv", 1);
        _objOfList = getOneHotList(Overlord.resourcesDir + "hist_objOf.csv", 1);
        _cardList = getOneHotList(Overlord.resourcesDir + "hist_cardinality.csv", 0);
        _pronounList = getOneHotList(Overlord.resourcesDir + "hist_pronoun.csv", 1);
        _pronounTypeList = getOneHotList(Overlord.resourcesDir + "hist_pronounType.csv", 0);

        _hypernyms = new ArrayList<>();
        for(String[] row : FileIO.readFile_table(Overlord.resourcesDir + "hist_hypernym.csv"))
            _hypernyms.add(row[0]);

        //read other from files
        _colors = new HashSet<>(FileIO.readFile_lineList(Overlord.resourcesDir + "colors.txt"));
        _stopWords = new HashSet<>(FileIO.readFile_lineList(Overlord.resourcesDir + "stop_words.txt"));
        _detList = new ArrayList<>();
        _detSet_singular = new HashSet<>();
        _detSet_plural = new HashSet<>();
        String[][] detTable = FileIO.readFile_table(Overlord.resourcesDir + "dets.csv");
        for (String[] row : detTable) {
            if (row.length > 1) {
                _detList.add(row[0]);
                if (row[1].equals("singular")) {
                    _detSet_singular.add(row[0]);
                } else if (row[1].equals("plural")) {
                    _detSet_plural.add(row[0]);
                }
            }
        }
    }

    private static Set<String> getSansIDs(Collection<Document> docSet, String sansType) {
        Set<String> ignoreIDs = new HashSet<>();
        for (Document d : docSet) {
            List<Mention> mentionList = d.getMentionList();
            for (int i = 0; i < mentionList.size(); i++) {
                for (int j = i + 1; j < mentionList.size(); j++) {
                    Mention m1 = mentionList.get(i);
                    Mention m2 = mentionList.get(j);

                    if (sansType.contains("nonvis"))
                        if (m1.getChainID().equals("0") || m2.getChainID().equals("0"))
                            ignoreIDs.add(Document.getMentionPairStr(m1, m2, true));
                    if (sansType.contains("pronom"))
                        if (m1.getPronounType() != Mention.PRONOUN_TYPE.NONE || m2.getPronounType() != Mention.PRONOUN_TYPE.NONE)
                            ignoreIDs.add(Document.getMentionPairStr(m1, m2, true));
                }
            }
        }
        return ignoreIDs;
    }

    public static void exportFeatures_pairwise(Collection<Document> docSet, String outroot, int numThreads) {
        _outroot = outroot + ".feats";

        Logger.log("Initializing lists");
        ClassifyUtil.initLists();

        Logger.log("Mapping mentions to their subjOf / objOf verbs");
        _subjOfDict = new HashMap<>();
        _objOfDict = new HashMap<>();
        for (Document d : docSet) {
            for (Caption c : d.getCaptionList()) {
                for (Mention m : c.getMentionList()) {
                    Chunk subjOf = c.getSubjectOf(m);
                    Chunk objOf = c.getObjectOf(m);
                    if (subjOf != null)
                        _subjOfDict.put(m, subjOf);
                    if (objOf != null)
                        _objOfDict.put(m, objOf);
                }
            }
        }

        Logger.log("Feature preprocessing (lemma counts)");
        _imgLemmaCountDict = new HashMap<>();
        for (Document d : docSet) {
            _imgLemmaCountDict.put(d.getID(), new DoubleDict<>());
            for (Mention m : d.getMentionList())
                _imgLemmaCountDict.get(d.getID()).increment(m.getHead().getLemma().toLowerCase().trim());
        }

        Logger.log("Feature preprocessing (only type)");
        _onlyTypeMentions = new HashSet<>();
        for (Document d : docSet) {
            for (Caption c : d.getCaptionList()) {
                //count the types being used, then set our only type field
                DoubleDict<String> typeCountDict = new DoubleDict<>();
                for (Mention m : c.getMentionList())
                    typeCountDict.increment(m.getLexicalType());
                for (Mention m : c.getMentionList())
                    if (typeCountDict.get(m.getLexicalType()) == 1)
                        _onlyTypeMentions.add(m);
            }
        }

        Logger.log("Feature preprocessing (adjecent prepositions)");
        _prepDict_left = new HashMap<>();
        _prepDict_right = new HashMap<>();
        for(Document d : docSet){
            for(Caption c : d.getCaptionList()){
                for(Mention m : c.getMentionList()){
                    List<Chunk> chunkList = m.getChunkList();
                    if(!m.getChunkList().isEmpty()){
                        Chunk left = c.getLeftNeighbor(chunkList.get(0));
                        Chunk right = c.getRightNeighbor(chunkList.get(chunkList.size()-1));
                        if(left != null && left.getChunkType().equals("PP"))
                            _prepDict_left.put(m, left.toString().toLowerCase());
                        if(right != null && right.getChunkType().equals("PP"))
                            _prepDict_right.put(m, right.toString().toLowerCase());
                    }
                }
            }
        }

        Logger.log("Feature preprocessing (chunk neighbors)");
        _mentionChunkNeighborDict = new HashMap<>();
        for (Document d : docSet) {
            for (Caption c : d.getCaptionList()) {
                List<Chunk> chunkList = c.getChunkList();
                List<Token> tokenList = c.getTokenList();
                for (Mention m : c.getMentionList()) {
                    int[] tokenRange = m.getTokenRange();
                    Chunk[] chunkNeighbors = {null, null};
                    Chunk startChunk = new Chunk(d.getID(), c.getIdx(), -1, "START", new ArrayList<>());
                    Chunk endChunk = new Chunk(d.getID(), c.getIdx(), -1, "END", new ArrayList<>());

                    //if the left or right token to this mention are outside the bounds, set the
                    //chunk appropriately
                    int chunkIdx_left = -1;
                    int chunkIdx_right = -1;
                    if (tokenRange[0] - 1 < 0) {
                        chunkNeighbors[0] = startChunk;
                    } else {
                        chunkIdx_left = tokenList.get(tokenRange[0] - 1).chunkIdx;
                    }
                    if (tokenRange[1] + 1 > tokenList.size() - 1) {
                        chunkNeighbors[1] = endChunk;
                    } else {
                        chunkIdx_right = tokenList.get(tokenRange[1] + 1).chunkIdx;
                        if (chunkIdx_right > chunkList.size() - 1) {
                            chunkNeighbors[1] = endChunk;
                            //set the chunk idx to -1, so it doesn't get updated
                            chunkIdx_right = -1;
                        }
                    }

                    //it's possible to reach here and have an invalid chunk idx, and
                    //in these cases we want to keep that initial null assignment
                    if (chunkIdx_left > -1)
                        chunkNeighbors[0] = chunkList.get(chunkIdx_left);
                    if (chunkIdx_right > -1)
                        chunkNeighbors[1] = chunkList.get(chunkIdx_right);

                    _mentionChunkNeighborDict.put(m, chunkNeighbors);
                }
            }
        }

        Logger.log("Feature preprocessing (hypernyms)");
        _hypDict = new HashMap<>();
        WordnetUtil wnUtil = new WordnetUtil(Overlord.wordnetDir);
        for(Document d : docSet){
            for(Mention m : d.getMentionList()){
                String lemma = m.getHead().getLemma().toLowerCase();
                if(!_hypDict.containsKey(lemma)){
                    Set<String> leaves = new HashSet<>();
                    HypTree tree = wnUtil.getHypernymTree(lemma);
                    for(List<HypTree.HypNode> branch : tree.getRootBranches()) {
                        String leaf = null;
                        for (HypTree.HypNode node : branch) {
                            if(_hypernyms.contains(node.toString())) {
                                leaf = node.toString();
                                break;
                            }
                        }
                        if(leaf != null)
                            leaves.add(leaf);
                    }
                    _hypDict.put(lemma, leaves);
                }
            }
        }

        exportFeatures(docSet, ExtractionThreadType.PAIRWISE, numThreads);

        JsonIO.writeFile(ExtractionThread.getMetaDict(), outroot + "_meta", false);
    }

    public static void exportFeatures_nonvis(Collection<Document> docSet, String outroot)
    {
        Logger.log("Feature preprocessing (hypernyms)");
        _hypDict = new HashMap<>();
        WordnetUtil wnUtil = new WordnetUtil(Overlord.wordnetDir);
        for(Document d : docSet){
            for(Mention m : d.getMentionList()){
                String lemma = m.getHead().getLemma().toLowerCase();
                if(!_hypDict.containsKey(lemma)){
                    Set<String> leaves = new HashSet<>();
                    HypTree tree = wnUtil.getHypernymTree(m.getHead().getLemma());
                    for(List<HypTree.HypNode> branch : tree.getRootBranches()) {
                        String leaf = null;
                        for (HypTree.HypNode node : branch) {
                            if(_hypernyms.contains(node.toString())) {
                                leaf = node.toString();
                                break;
                            }
                        }
                        if(leaf != null)
                            leaves.add(leaf);
                    }
                    _hypDict.put(lemma, leaves);
                }
            }
        }

        Logger.log("Feature preprocessing (subj/obj of verbs)");
        _subjOfDict = new HashMap<>();
        _objOfDict = new HashMap<>();
        for (Document d : docSet) {
            for (Caption c : d.getCaptionList()) {
                for (Mention m : c.getMentionList()) {
                    Chunk subjOf = c.getSubjectOf(m);
                    Chunk objOf = c.getObjectOf(m);
                    if (subjOf != null)
                        _subjOfDict.put(m, subjOf);
                    if (objOf != null)
                        _objOfDict.put(m, objOf);
                }
            }
        }

        Logger.log("Feature preprocessing (adjecent prepositions)");
        _prepDict_left = new HashMap<>();
        _prepDict_right = new HashMap<>();
        for(Document d : docSet){
            for(Caption c : d.getCaptionList()){
                for(Mention m : c.getMentionList()){
                    List<Chunk> chunkList = m.getChunkList();
                    if(!m.getChunkList().isEmpty()){
                        Chunk left = c.getLeftNeighbor(chunkList.get(0));
                        Chunk right = c.getRightNeighbor(chunkList.get(chunkList.size()-1));
                        if(left != null && left.getChunkType().equals("PP"))
                            _prepDict_left.put(m, left.toString().toLowerCase());
                        if(right != null && right.getChunkType().equals("PP"))
                            _prepDict_right.put(m, right.toString().toLowerCase());
                    }
                }
            }
        }

        Map<String, Object> metaDict = new HashMap<>();
        int start = 1, end = start + _headList.size();
        metaDict.put("head", new Integer[]{start, end});
        start = end + 1; end = start + _numericList.size();
        metaDict.put("numeric", new Integer[]{start, end});
        start = end + 1; end = start + _modifierList.size();
        metaDict.put("modifier", new Integer[]{start, end});
        start = end + 1; end = start + _subjOfList.size();
        metaDict.put("subj_of", new Integer[]{start, end});
        start = end + 1; end = start + _objOfList.size();
        metaDict.put("obj_of", new Integer[]{start, end});
        start = end + 1; end = start + _prepositionList.size();
        metaDict.put("preposition_left", new Integer[]{start, end});
        start = end + 1; end = start + _prepositionList.size();
        metaDict.put("preposition_right", new Integer[]{start, end});
        start = end + 1; end = start + _hypernyms.size();
        metaDict.put("hypernym", new Integer[]{start, end});
        metaDict.put("max_idx", end+1);

        Logger.log("Extracting features");
        Set<FeatureVector> fvSet = new HashSet<>();
        for(Document d : docSet){
            for(Mention m : d.getMentionList()){
                if(m.getPronounType() != Mention.PRONOUN_TYPE.NONE &&
                   m.getPronounType() != Mention.PRONOUN_TYPE.SEMI)
                    continue;

                FeatureVector fv = new FeatureVector();
                fv.comments = m.getUniqueID();
                fv.label = m.getChainID().equals("0") ? 1.0 : 0.0;
                int currentIdx = 0;

                String head = m.getHead().toString().toLowerCase();
                if(_headList.indexOf(head) > -1)
                    fv.addFeature(currentIdx + _headList.indexOf(head) + 1, 1.0);
                currentIdx += _headList.size() + 1;

                String[] mods = m.getModifiers();
                if(_numericList.indexOf(mods[0]) > -1)
                    fv.addFeature(currentIdx + _numericList.indexOf(mods[0]) + 1, 1.0);
                currentIdx += _numericList.size() + 1;
                if(_modifierList.indexOf(mods[1]) > -1)
                    fv.addFeature(currentIdx + _modifierList.indexOf(mods[1]) + 1, 1.0);
                currentIdx += _modifierList.size() + 1;

                Chunk subjOf = _subjOfDict.get(m);
                String subjOfStr = "";
                if(subjOf != null)
                    subjOfStr = subjOf.getTokenList().get(subjOf.getTokenList().size()-1).toString().toLowerCase();
                if(_subjOfList.indexOf(subjOfStr) > -1)
                    fv.addFeature(currentIdx + _subjOfList.indexOf(subjOfStr) + 1, 1.0);
                currentIdx += _subjOfList.size() + 1;

                Chunk objOf = _objOfDict.get(m);
                String objOfStr = "";
                if(objOf != null)
                    objOfStr = objOf.getTokenList().get(objOf.getTokenList().size()-1).toString().toLowerCase();
                if(_objOfList.indexOf(objOfStr) > -1)
                    fv.addFeature(currentIdx + _objOfList.indexOf(objOfStr) + 1, 1.0);
                currentIdx += _objOfList.size() + 1;

                String left_prep = _prepDict_left.get(m);
                if(left_prep != null && _prepositionList.indexOf(left_prep) > -1)
                    fv.addFeature(currentIdx + _prepositionList.indexOf(left_prep) + 1, 1.0);
                currentIdx += _prepositionList.size() + 1;
                String right_prep = _prepDict_right.get(m);
                if(right_prep != null && _prepositionList.indexOf(right_prep) > -1)
                    fv.addFeature(currentIdx + _prepositionList.indexOf(right_prep) + 1, 1.0);
                currentIdx += _prepositionList.size() + 1;

                Set<String> hypSet = _hypDict.get(m.getHead().getLemma().toLowerCase());
                for(String hyp : _hypernyms){
                    if(hypSet != null && hypSet.contains(hyp))
                        fv.addFeature(currentIdx, 1.0);
                    currentIdx++;
                }
                fvSet.add(fv);
            }
        }

        Logger.log("Writing to " + outroot + ".feats");
        FileIO.writeFile(fvSet, outroot, "feats", false);
        JsonIO.writeFile(metaDict, outroot + "_meta", false);
    }

    private int addOneHotVector(FeatureVector fv, int currentIdx,
                                List<String> oneHotList, String item)
    {
        if(oneHotList.indexOf(item) > -1)
            fv.addFeature(currentIdx + oneHotList.indexOf(item) + 1, 1.0);
        return currentIdx + oneHotList.size() + 1;
    }

    private static List<Double> getMentionVector(String compType, Mention m) {
        List<Double> emptyVec = new ArrayList<>();
        for (int i = 0; i < 300; i++)
            emptyVec.add(0.0);

        Set<String> ignoredPOS = new HashSet<>();
        ignoredPOS.add("DT");
        ignoredPOS.add("CD");
        ignoredPOS.add("PRP");
        ignoredPOS.add("PRP$");

        List<Double> vec = new ArrayList<>(emptyVec);
        if (m.getPronounType() == Mention.PRONOUN_TYPE.NONE) {
            List<List<Double>> vecList = new ArrayList<>();
            for (Token t : m.getTokenList()) {
                if (!ignoredPOS.contains(t.getPosTag())) {
                    String word = StringUtil.keepAlpha(t.toString()).toLowerCase();
                    if (!word.isEmpty())
                        vecList.add(_w2vUtil.getVector(word));
                }
            }
            if (vecList.size() > 0) {
                switch (compType) {
                    case "add":
                        vec = Util.vectorAdd(vecList);
                        break;
                    case "mean":
                        vec = Util.vectorMean(vecList);
                        break;
                    case "mult":
                        vec = Util.vectorMean(vecList);
                        break;
                }
            }
        }
        return vec;
    }

    /**
     * Returns a mapping of document IDs to a set of mention pairs,
     * where the second mention in the pair is a pronominal, and
     * the first is the antecedent with which it's coreferent
     *
     * @param docSet
     * @return
     */
    @Deprecated
    public static Map<String, Set<Mention[]>> _pronominalCoref(Collection<Document> docSet) {
        Map<String, Set<Mention[]>> imgMentionPairDict = new HashMap<>();
        WordnetUtil wnUtil = new WordnetUtil(Overlord.wordnetDir);

        for (Document d : docSet) {
            for (Caption c : d.getCaptionList()) {
                //start at the second mention, since first mentions cannot have valid antecedents
                for (int i = 1; i < c.getMentionList().size(); i++) {
                    Mention m_pronom = c.getMentionList().get(i);
                    Mention.PRONOUN_TYPE pronounType = m_pronom.getPronounType();

                    if (pronounType != Mention.PRONOUN_TYPE.NONE &&
                            pronounType != Mention.PRONOUN_TYPE.INDEFINITE) {
                        /** We want to assign coreference based on pronoun type,
                         *  according to the below heuristics
                         *
                         *  1) Subjective / Objective / Special ("it")
                         *     - Attach to furthest antecedent; usually main subject
                         *  2) Reflexive / Reciprocal
                         *     - Attach to the nearest antecedent; usually in same clause
                         *  3) Relative / Special ("that")
                         *     - If adjacent left neighbor is candidate, attach (NP [who/that])
                         *     - If XofY construction, exclude X, attach to nearest
                         *     - Otherwise attach nearest antecedent
                         *  4) Indefinite
                         *     - No attachment
                         *  5) Other
                         *     - "another", "other", "others"; no attachment
                         *     - "both"; attach to a animate, 2-card antecedent
                         *     - "one"; no attachment
                         * */

                        //collect left-to-right candidate antecedents
                        List<Mention> candAnteList = new ArrayList<>();
                        List<Mention> candAnteList_plural = new ArrayList<>();
                        for (int j = i - 1; j >= 0; j--) {
                            Mention m_ante = c.getMentionList().get(j);
                            /* A mention may only be an agent pronoun antecedent if
                             *   a) The cardinality of it and the pronoun are approximately equal
                             *   b) The gender of it and the pronoun match
                             */
                            Boolean pluralPronom =
                                    Mention.PRONOUN_TYPE.getIsPlural(m_pronom.toString().toLowerCase());
                            boolean cardMatch = false;
                            if (pluralPronom != null) {
                                if (pluralPronom && m_ante.getCardinality().getValue() > 1) {
                                    cardMatch = true;
                                } else if (!pluralPronom && m_ante.getCardinality().getValue() == 1) {
                                    cardMatch = true;
                                }
                            }
                            if (cardMatch) {
                                //we check gender last, since getting the hypernyms for
                                //the antecedent is more costly than the other steps
                                boolean genderMatch = getGenderMatch(m_pronom, m_ante);
                                if (genderMatch) {
                                    candAnteList.add(m_ante);
                                    if (m_ante.getCardinality().getValue() > 1) {
                                        candAnteList_plural.add(m_ante);
                                    }
                                }
                            }

                            //if we've collected plural candidates, that means
                            //that our pronoun is ambiguous (ie. they); in these cases,
                            //we should prefer plural antecedents to singular ones,
                            //which in this case means dropping singular antecedents
                            /*
                            if(!candAnteList_plural.isEmpty())
                                candAnteList = candAnteList_plural;
                            */
                        }


                        //Determine coreference between mentions, as an {ante, pronom} pair
                        Mention[] mentionPair = new Mention[2];
                        mentionPair[1] = m_pronom;
                        if (!candAnteList.isEmpty()) {
                             /* 1) Subjective / Objective */
                            if (pronounType == Mention.PRONOUN_TYPE.SUBJECTIVE_SINGULAR ||
                                    pronounType == Mention.PRONOUN_TYPE.SUBJECTIVE_PLURAL ||
                                    pronounType == Mention.PRONOUN_TYPE.OBJECTIVE_SINGULAR ||
                                    pronounType == Mention.PRONOUN_TYPE.OBJECTIVE_PLURAL ||
                                    m_pronom.toString().toLowerCase().equals("it")) {
                                mentionPair[0] = candAnteList.get(candAnteList.size() - 1);
                            }
                            /* 2) Reflexive */
                            else if (pronounType == Mention.PRONOUN_TYPE.REFLEXIVE_SINGULAR ||
                                    pronounType == Mention.PRONOUN_TYPE.REFLEXIVE_PLURAL ||
                                    pronounType == Mention.PRONOUN_TYPE.RECIPROCAL) {
                                mentionPair[0] = candAnteList.get(0);
                            }
                            /* 3) Relative */
                            else if (pronounType == Mention.PRONOUN_TYPE.RELATIVE ||
                                    m_pronom.toString().toLowerCase().equals("that")) {
                                String normText = m_pronom.toString().toLowerCase().trim();
                                List<Token> interstitialTokens =
                                        c.getTokenList().subList(candAnteList.get(0).getTokenRange()[1],
                                                m_pronom.getTokenRange()[0] + 1);
                                String interstitialStr = StringUtil.listToString(interstitialTokens, " ");

                                //If adjacent left neighbor is a candidate; attach
                                if (candAnteList.get(0).getTokenRange()[1] + 1 ==
                                        m_pronom.getTokenRange()[0]) {
                                    mentionPair[0] = candAnteList.get(0);
                                }
                                //If pronoun is Y in XofY, exclude X, attach to nearest
                                else if (interstitialStr.equals("of")) {
                                    candAnteList.remove(0);
                                    if (!candAnteList.isEmpty())
                                        mentionPair[0] = candAnteList.get(0);
                                }
                                //Otherwise, attach to nearest
                                else {
                                    mentionPair[0] = candAnteList.get(0);
                                }
                            } else if (pronounType == Mention.PRONOUN_TYPE.OTHER) {
                                if (m_pronom.toString().toLowerCase().equals("both")) {
                                    for (int k = 0; k < candAnteList.size(); k++) {
                                        Mention candAnte = candAnteList.get(k);
                                        if (candAnte.getCardinality().getValue() == 2 &&
                                                !candAnte.getCardinality().isUnderdef()) {
                                            mentionPair[0] = candAnteList.get(k);
                                            break;
                                        }
                                    }
                                }
                            }

                            /* C7) ignore candidate pronouns with
                             *     multiple candidate antecedents
                             *     that appear in a [NP ... and ... NP ...] structure
                             */
                            /*
                            String pattern_list = "^.*(NP (PP |NP |ADJP )*(, )?)+(and (NP)).*$";
                            if(candAnteList.size() > 1 && c.toChunkTypeString(true).matches(pattern_list)){
                                continue;
                            }*/
                        }
                        //if we haven't set the antecedent; log this caption
                        if (mentionPair[0] != null) {
                            if (!imgMentionPairDict.containsKey(d.getID()))
                                imgMentionPairDict.put(d.getID(), new HashSet<>());
                            imgMentionPairDict.get(d.getID()).add(mentionPair);
                        }
                    }
                }
            }
        }
        return imgMentionPairDict;
    }

    /**
     * Performs intra-caption pronominal subset prediction (pronominal and not),
     * returning a mapping of [docID -> {(super, sub) ...}]
     *
     * @param docSet
     * @return
     */
    @Deprecated
    public static Map<String, Set<Mention[]>> pronomSubset(Collection<Document> docSet) {
        Map<String, Set<Mention[]>> subsetDict = new HashMap<>();
        for (Document d : docSet) {
            Set<Mention[]> subsets = new HashSet<>();
            for (Caption c : d.getCaptionList()) {
                for (int i = 0; i < c.getMentionList().size(); i++) {
                    Mention m1 = c.getMentionList().get(i);
                    Mention[] supSub = null;
                    for (int j = i - 1; j >= 0; j--) {
                        Mention m2 = c.getMentionList().get(j);

                        /* In the pronominal case, we're interested in one of two
                         * conditions
                         *      a) m1 is (non-indef) pronom; antecedent is a superset
                         *      b) m1 is (non-indef) pronom; antecedent is a subset
                         */
                        if (m1.getPronounType() != Mention.PRONOUN_TYPE.NONE &&
                                m1.getPronounType() != Mention.PRONOUN_TYPE.INDEFINITE &&
                                m2.getPronounType() == Mention.PRONOUN_TYPE.NONE) {
                            Cardinality c1 = m1.getCardinality();
                            Cardinality c2 = m2.getCardinality();

                            //We dictate which is a subset of which based on
                            //cardinality.
                            //1 sub 2 (unambig); 2 sub 1 unambig
                            if (!c1.isUnderdef() && !c2.isUnderdef()) {
                                if (c1.getValue() < c2.getValue())
                                    supSub = new Mention[]{m2, m1};
                                else if (c2.getValue() < c1.getValue())
                                    supSub = new Mention[]{m1, m2};
                            } //2 sub 1
                            else if (c1.isUnderdef() && !c2.isUnderdef()) {
                                supSub = new Mention[]{m1, m2};
                            } //1 sub 2
                            else if (!c1.isUnderdef() && c2.isUnderdef()) {
                                supSub = new Mention[]{m2, m1};
                            }
                        }
                    }

                    if (supSub != null)
                        subsets.add(supSub);
                }
            }
            if (!subsets.isEmpty())
                subsetDict.put(d.getID(), subsets);
        }
        return subsetDict;
    }


    /**Evaluates the pronominal coreference system, where the score dict
     * is computed over all the links where at least one mention is
     * fully pronominal
     *
     * @param docSet
     * @return
     */
    public static ScoreDict<Integer> evaluatePronomCoref(Collection<Document> docSet) {
        ScoreDict<Integer> scores = new ScoreDict<>();
        for(Document d : docSet){
            Map<Mention[], String> pronomCases = Minion.filter_pronomCorefEval(d);
            for(Mention[] pair : pronomCases.keySet()){
                String caseStr = pronomCases.get(pair);
                int gold = caseStr.contains("gold") ? 1 : 0;
                int pred = caseStr.contains("pred") ? 1 : 0;
                scores.increment(gold, pred);
            }
        }
        return scores;
    }

    public static Set<String> pronominalCoref(Document d)
    {
        Set<String> corefPairs = new HashSet<>();
        for (Caption c : d.getCaptionList()) {

            //start at the second mention, since we're only concerned with
            //pronominal mentions with non-pronominal antecedents
            for (int i = 1; i < c.getMentionList().size(); i++) {
                Mention m_pronom = c.getMentionList().get(i);
                Mention.PRONOUN_TYPE pronounType = m_pronom.getPronounType();

                //both subset and coreference relations only occur between
                //non-indefinite pronouns
                if (pronounType != Mention.PRONOUN_TYPE.NONE &&
                    pronounType != Mention.PRONOUN_TYPE.INDEFINITE &&
                    pronounType != Mention.PRONOUN_TYPE.SEMI) {

                    //collect left-to-right candidate antecedents
                    List<Mention> candAnteList = new ArrayList<>();
                    for (int j = i - 1; j >= 0; j--) {
                        Mention m_ante = c.getMentionList().get(j);
                        if (m_ante.getPronounType() == Mention.PRONOUN_TYPE.NONE)
                            candAnteList.add(m_ante);
                    }
                    if (candAnteList.isEmpty())
                        continue;   //we can't attach anything that has no antecedents

                    //Determine a relation, either as
                    //  {ante, pronom} pair (coref)
                    //  {sub, sup} pair (subset)
                    Mention[] pair = null;

                    /** We want to assign coreference based on pronoun type,
                     *  according to the below heuristics
                     *
                     *  1) Subjective / Objective / Special ("it")
                     *     - Attach to furthest antecedent; usually main subject
                     *  2) Reflexive / Reciprocal
                     *     - Attach to the nearest antecedent; usually in same clause
                     *  3) Relative / Special ("that")
                     *     - If adjacent left neighbor is candidate, attach (NP [who/that])
                     *     - If XofY construction, exclude X, attach to nearest
                     *     - Otherwise attach nearest antecedent
                     *  4) Indefinite
                     *     - No attachment
                     *  5) Other
                     *     - "another", "other", "others"; no attachment
                     *     - "both"; attach to a animate, 2-card antecedent
                     *     - "one"; no attachment
                     */

                    //Retain only the antecedents that have the correct numerical
                    //modifier
                    List<Mention> candAnteList_coref = new ArrayList<>();
                    for (Mention m : candAnteList) {
                        Boolean pluralPronom = Mention.PRONOUN_TYPE.getIsPlural(m_pronom.toString().toLowerCase());
                        if (pluralPronom != null) {
                            /* //Removing cardinality
                            if ((pluralPronom && m.getCardinality().getValue() > 1) ||
                                (!pluralPronom && m.getCardinality().getValue() == 1)) {
                                if (getGenderMatch(m_pronom, m))
                                    candAnteList_coref.add(m);
                            }*/
                            String[] singMods = {"one", "1", "first", "second", "third"};

                            String[] mods = m.getModifiers();
                            boolean pluralAnte = false;
                            if(m.getHead().getPosTag().equals("NNS") ||
                               m.getHead().getPosTag().equals("NNPS")){
                                pluralAnte = true;
                            } else if(!mods[0].isEmpty() &&
                                      !Arrays.asList(singMods).contains(mods[0])) {
                                pluralAnte = true;
                            }

                            //if the pronoun is which/who/that/them,
                            //attach everything regardless of plurality
                            String[] specialPronouns = {"which", "who", "that"};
                            if(Arrays.asList(specialPronouns).contains(m_pronom.toString().toLowerCase())) {
                                candAnteList_coref.add(m);
                            } else if (pluralPronom && pluralAnte || !pluralPronom && !pluralAnte) {
                                if (getGenderMatch(m_pronom, m))
                                    candAnteList_coref.add(m);
                            }
                        }
                    }

                    //
                    int lastIdx = candAnteList_coref.size() - 1;
                    if (!candAnteList_coref.isEmpty()) {
                        if(m_pronom.toString().equalsIgnoreCase("it")){
                            pair = new Mention[]{candAnteList_coref.get(0), m_pronom};
                        } /* 1) Subjective / Objective */
                        else if (pronounType == Mention.PRONOUN_TYPE.SUBJECTIVE_SINGULAR ||
                            pronounType == Mention.PRONOUN_TYPE.SUBJECTIVE_PLURAL ||
                            pronounType == Mention.PRONOUN_TYPE.OBJECTIVE_SINGULAR ||
                            pronounType == Mention.PRONOUN_TYPE.OBJECTIVE_PLURAL) {

                            Mention ante = candAnteList_coref.get(lastIdx);
                            //Prefer people/animal pronouns, where available
                            for(int j=lastIdx; j>=0; j--){
                                Mention m_j = candAnteList_coref.get(j);
                                if(m_j.getLexicalType().contains("people") ||
                                   m_j.getLexicalType().contains("animals")){
                                    ante = candAnteList_coref.get(j);
                                    break;
                                }
                            }
                            pair = new Mention[]{ante, m_pronom};
                        } /* 2) Reflexive */
                        else if (pronounType == Mention.PRONOUN_TYPE.REFLEXIVE_SINGULAR ||
                                 pronounType == Mention.PRONOUN_TYPE.REFLEXIVE_PLURAL ||
                                 pronounType == Mention.PRONOUN_TYPE.RECIPROCAL) {

                            Mention ante = candAnteList_coref.get(0);
                            //Prefer people/animal pronouns, where available
                            for(int j=0; j<=lastIdx; j++){
                                Mention m_j = candAnteList_coref.get(j);
                                if(m_j.getLexicalType().contains("people") ||
                                   m_j.getLexicalType().contains("animals")){
                                    ante = candAnteList_coref.get(j);
                                    break;
                                }
                            }
                            pair = new Mention[]{ante, m_pronom};
                        } /* 3) Relative */
                        else if (pronounType == Mention.PRONOUN_TYPE.RELATIVE ||
                                m_pronom.toString().toLowerCase().equals("that")) {
                            List<Token> interstitialTokens =
                                    c.getTokenList().subList(candAnteList_coref.get(0).getTokenRange()[1],
                                            m_pronom.getTokenRange()[0] + 1);
                            String interstitialStr = StringUtil.listToString(interstitialTokens, " ");

                            //If adjacent left neighbor is a candidate; attach
                            if (candAnteList_coref.get(0).getTokenRange()[1] + 1 ==
                                    m_pronom.getTokenRange()[0]) {
                                pair = new Mention[]{candAnteList_coref.get(0), m_pronom};
                            }
                            //If pronoun is Y in XofY, exclude X, attach to nearest
                            else if (interstitialStr.equals("of")) {
                                candAnteList_coref.remove(0);
                                if (!candAnteList_coref.isEmpty())
                                    pair = new Mention[]{candAnteList_coref.get(0), m_pronom};
                            }
                            //Otherwise, attach to nearest
                            else {
                                pair = new Mention[]{candAnteList_coref.get(0), m_pronom};
                            }
                        } /* 4) Other */ else if (pronounType == Mention.PRONOUN_TYPE.OTHER) {
                            if (m_pronom.toString().toLowerCase().equals("both")) {
                                for (int k = 0; k < candAnteList_coref.size(); k++) {
                                    Mention candAnte = candAnteList_coref.get(k);
                                    String[] mods = candAnte.getModifiers();
                                    if(mods[0].equals("two") || mods[0].equals("2"))
                                        pair = new Mention[]{candAnte, m_pronom};
                                }
                            }
                        }
                    }

                    if(pair != null){
                        corefPairs.add(Document.getMentionPairStr(pair[0], pair[1], true, true));
                        corefPairs.add(Document.getMentionPairStr(pair[1], pair[0], true, true));
                    }
                }
            }
        }
        return corefPairs;
    }

    /**
     * Performs attribute attachment, associating animate mentions (as keys) with
     * attributes in the form of Annotation objects (Mentions, Chunks, or Tokens)
     *
     * @param docSet
     * @return
     */
    public static Map<Mention, AttrStruct> attributeAttachment_agent(Collection<Document> docSet) {
        Map<Mention, AttrStruct> attributeDict = new HashMap<>();

        String[][] clothAttrLex = FileIO.readFile_table(Overlord.resourcesDir + "clothAttrLex.csv");
        _clothAttrLex = new HashMap<>();
        for (String[] row : clothAttrLex)
            _clothAttrLex.put(row[0], row[1]);

        Logger.log("Storing mention genders");
        Map<Mention, String> genderDict = new HashMap<>();
        /*
        for (Document d : docSet) {
            for (Mention m : d.getMentionList()) {
                List<String> hyps = getHypernyms(m.getHead().getLemma());
                String gender = m.getGender(hyps);
                if (!gender.equals("neuter"))
                    genderDict.put(m, gender);
            }
        }*/

        Logger.log("Associating bodyparts with agent mentions");
        for (Document d : docSet) {
            for (Caption c : d.getCaptionList()) {
                List<Mention> bodypartList = new ArrayList<>();
                List<Mention> agentList = new ArrayList<>();
                for (Mention m : c.getMentionList()) {
                    String normText = " " + m.toString().toLowerCase().trim();
                    if (m.getLexicalType().equals("people") ||
                            m.getLexicalType().equals("animals")) {
                        agentList.add(m);
                    } else if (m.getPronounType().isAnimate()) {
                        agentList.add(m);
                    } else if (normText.endsWith(" one") ||
                            normText.endsWith(" other") ||
                            normText.endsWith(" another") ||
                            normText.endsWith(" others")) {
                        agentList.add(m);
                    } else if (m.getLexicalType().equals("bodyparts")) {
                        bodypartList.add(m);
                    }
                }

                if (!agentList.isEmpty() && !bodypartList.isEmpty()) {
                    //Given the crudeness of this heuristic, we need to cluster
                    //agent mentions together (ex. "A man and a woman waving their arms")
                    List<List<Mention>> clusterList_agent =
                            collapseMentionListToConstructionList(agentList, c);

                    for (Mention m_parts : bodypartList) {
                        String gender_parts = "neuter";
                        if (genderDict.containsKey(m_parts))
                            gender_parts = genderDict.get(m_parts);

                        //Find the nearest agent antecedent cluster;
                        //if its a singleton cluster, check the gender as well
                        List<Mention> nearestAgentCluster_left = null;
                        List<Mention> nearestAgentCluster_right = null;
                        int maxIdx = -1;
                        int minIdx = -1;
                        for (List<Mention> agentCluster : clusterList_agent) {
                            //for the purposes of determining the nearest
                            //mention, get the last in the cluster
                            Mention m_last = agentCluster.get(agentCluster.size() - 1);
                            int idx = m_last.getIdx();

                            //treat all multi-element clusters as genderless
                            String gender_agent = "neuter";
                            if (clusterList_agent.size() == 1 && genderDict.containsKey(m_last))
                                gender_agent = genderDict.get(m_last);

                            //store the agent clusters that are nearest on the left
                            //and right of the bodypart mention
                            boolean genderMatch = gender_parts.equals("neuter") ||
                                    gender_agent.equals("neuter") ||
                                    gender_parts.equals(gender_agent);
                            if (idx < m_parts.getIdx() && idx > maxIdx && genderMatch) {
                                maxIdx = idx;
                                nearestAgentCluster_left = agentCluster;
                            } else if (idx > m_parts.getIdx() && idx < minIdx && genderMatch) {
                                minIdx = idx;
                                nearestAgentCluster_right = agentCluster;
                            }
                        }

                        List<Mention> agentCluster = null;
                        //1) Associate the nearest following agent cluster if
                        //   in an parts of people construction (ie "the arm of a man")
                        String interstitialString_right = "";
                        if (nearestAgentCluster_right != null) {
                            Mention m_pers = nearestAgentCluster_right.get(0);
                            interstitialString_right =
                                    StringUtil.listToString(c.getInterstitialChunks(m_parts, m_pers), " ");
                            interstitialString_right = interstitialString_right.toLowerCase().trim();
                            if (interstitialString_right.equals("of"))
                                agentCluster = nearestAgentCluster_right;
                        }
                        //2) Associate the nearest preceding agent cluster in all other cases
                        if (agentCluster == null && nearestAgentCluster_left != null)
                            agentCluster = nearestAgentCluster_left;

                        if (agentCluster != null) {
                            //invert the assciations, since we want to produce
                            //attributes for agents
                            for (Mention agent : agentCluster) {
                                if (!attributeDict.containsKey(agent))
                                    attributeDict.put(agent, toAttrStruct(agent));

                                String normText = m_parts.toString().toLowerCase();
                                String attrName = "bodypart";
                                if (normText.contains("head") || normText.contains("hair") || normText.contains("mouth") || normText.contains("face")) {
                                    attrName = "head";
                                } else if (normText.contains("chest")) {
                                    attrName = "torso";
                                } else if (normText.contains("arm")) {
                                    attrName = "arms";
                                } else if (normText.contains("leg")) {
                                    attrName = "legs";
                                } else if (normText.contains("hand")) {
                                    attrName = "hands";
                                } else if (normText.contains("foot") || normText.contains("feet")) {
                                    attrName = "feet";
                                }

                                if (normText.contains("her ")) {
                                    attributeDict.get(agent).clearAttribute("gender");
                                    attributeDict.get(agent).addAttribute("gender", "female");
                                } else if (normText.contains("his ")) {
                                    attributeDict.get(agent).clearAttribute("gender");
                                    attributeDict.get(agent).addAttribute("gender", "male");
                                }
                                attributeDict.get(agent).addAttribute(attrName, toAttrStruct(m_parts));
                            }
                        }
                    }
                }
            }
        }

        Logger.log("Associating clothing/colors with agent mentions");
        for (Document d : docSet) {
            for (Caption c : d.getCaptionList()) {
                //populate the lists of people and clothing
                List<Mention> clothingList = new ArrayList<>();
                List<Mention> agentList = new ArrayList<>();
                for (Mention m : c.getMentionList()) {
                    String normText = " " + m.toString().toLowerCase().trim();
                    if (m.getLexicalType().equals("people") ||
                            m.getLexicalType().equals("animals")) {
                        agentList.add(m);
                    } else if (m.getPronounType().isAnimate()) {
                        agentList.add(m);
                    } else if (normText.endsWith(" one") ||
                            normText.endsWith(" other") ||
                            normText.endsWith(" another") ||
                            normText.endsWith(" others")) {
                        agentList.add(m);
                    } else if (m.getLexicalType().equals("clothing")) {
                        clothingList.add(m);
                    } else if (m.getLexicalType().equals("colots")) {
                        clothingList.add(m);
                    }
                }

                if (!agentList.isEmpty() && !clothingList.isEmpty()) {
                    //put both clothing and agents into clusters, if they're in list format
                    List<List<Mention>> clusterList_cloth =
                            collapseMentionListToConstructionList(clothingList, c);
                    List<List<Mention>> clusterList_agent =
                            collapseMentionListToConstructionList(agentList, c);

                    //for each clothing cluster, associate the nearest agent
                    for (List<Mention> clothCluster : clusterList_cloth) {
                        //get the earliest clothing mention in the cluster
                        Mention clothMention = clothCluster.get(0);

                        //get the nearest agent cluster
                        List<Mention> nearestAgentCluster = new ArrayList<>();
                        int maxIdx = -1;
                        for (List<Mention> agentCluster : clusterList_agent) {
                            Mention lastAgentMention = agentCluster.get(agentCluster.size() - 1);
                            if (lastAgentMention.getIdx() < clothMention.getIdx() &&
                                    lastAgentMention.getIdx() > maxIdx) {
                                maxIdx = lastAgentMention.getIdx();
                                nearestAgentCluster = agentCluster;
                            }
                        }

                        //associate the nearest agent with this clothing cluster
                        for (Mention agent : nearestAgentCluster) {
                            if (!attributeDict.containsKey(agent))
                                attributeDict.put(agent, toAttrStruct(agent));
                            for (Mention cm : clothCluster) {
                                String clothNormText = cm.getHead().getLemma().toLowerCase();
                                String attrName = "clothing";
                                if (_clothAttrLex.containsKey(clothNormText))
                                    attrName = _clothAttrLex.get(clothNormText);
                                attributeDict.get(agent).addAttribute(attrName, toAttrStruct(cm));


                                if (clothNormText.contains("her ")) {
                                    attributeDict.get(agent).clearAttribute("gender");
                                    attributeDict.get(agent).addAttribute("gender", "female");
                                } else if (clothNormText.contains("his ")) {
                                    attributeDict.get(agent).clearAttribute("gender");
                                    attributeDict.get(agent).addAttribute("gender", "male");
                                }
                            }
                        }
                    }
                }
            }
        }

        Logger.log("Associating postmodifiers with agent mentions");
        for (Document d : docSet) {
            for (Caption c : d.getCaptionList()) {
                //if thiere's an immediately adjacent ADJP chunk to
                //a mention, attach it as an attribute
                for (Mention m : c.getMentionList()) {
                    if (!m.getChunkList().isEmpty()) {
                        int lastTokenIdx = m.getTokenRange()[1];
                        int lastChunkIdx = m.getChunkList().get(m.getChunkList().size() - 1).getIdx();
                        if (c.getChunkList().size() > lastChunkIdx + 1) {
                            Chunk nextChunk = c.getChunkList().get(lastChunkIdx + 1);
                            if (lastTokenIdx + 1 == nextChunk.getTokenRange()[0] &&
                                    nextChunk.getChunkType().equals("ADJP")) {
                                if (!attributeDict.containsKey(m))
                                    attributeDict.put(m, toAttrStruct(m));
                                String attrName = "modifier";
                                if (_colors.contains(nextChunk.toString().toLowerCase()))
                                    attrName = "color";
                                attributeDict.get(m).addAttribute(attrName, toAttrStruct(nextChunk));
                            }
                        }
                    }
                }
            }
        }

        return attributeDict;
    }

    private static AttrStruct toAttrStruct(Annotation core) {
        AttrStruct attr = new AttrStruct(core);
        String text = null;
        List<Token> ownedTokens = new ArrayList<>();
        if (core instanceof Token) {
            Token t = (Token) core;
            text = t.toString();
        } else if (core instanceof Chunk) {
            Chunk ch = (Chunk) core;
            text = ch.toString();
            ownedTokens = ch.getTokenList();
        } else if (core instanceof Mention) {
            Mention m = (Mention) core;
            text = m.toString();
            attr.addAttribute("cardinality", m.getCardinality().toString().replace("|", ","));
            attr.addAttribute("lexical_type", m.getLexicalType());
            attr.addAttribute("head_lemma", m.getHead().getLemma());
            if (m.getLexicalType().contains("people"))
                attr.addAttribute("gender", m.getGender());
            ownedTokens = m.getTokenList();
        }
        if (text != null)
            attr.addAttribute("text", text);
        for (Token t : ownedTokens) {
            String tokenText = t.toString().toLowerCase();
            if (_colors.contains(tokenText))
                attr.addAttribute("color", tokenText);
        }

        return attr;
    }

    private static List<List<Mention>>
    collapseMentionListToConstructionList(List<Mention> mentionList, Caption caption) {
        String[] allowedConjArr = {",", "and", ", and", "on and"};
        List<String> allowedConjList = Arrays.asList(allowedConjArr);

        List<List<Mention>> constructionList = new ArrayList<>();
        for (int i = 0; i < mentionList.size(); i++) {
            Mention currentMention = mentionList.get(i);

            //determine if the current construction needs to be closed out
            //prior to adding this mention
            String interstitialText = "";
            if (i > 0) {
                Mention prevMention = mentionList.get(i - 1);
                interstitialText = StringUtil.listToString(
                        caption.getInterstitialChunks(prevMention, currentMention), " ");
                interstitialText = interstitialText.toLowerCase().trim();
            }
            if (allowedConjList.contains(interstitialText)) {
                constructionList.get(constructionList.size() - 1).add(currentMention);
            } else {
                List<Mention> construction = new ArrayList<>();
                construction.add(currentMention);
                constructionList.add(construction);
            }
        }

        return constructionList;
    }

    private static int getChunkTypeMatch(String chunkType_1, String chunkType_2) {
        int typeMatch = UNK;
        if (chunkType_1 == null && chunkType_2 == null)
            typeMatch = TRUE;
        else if (chunkType_1 == null && chunkType_2 != null)
            typeMatch = FALSE;
        else if (chunkType_1 != null && chunkType_2 == null)
            typeMatch = FALSE;
        else
            typeMatch = chunkType_1.equals(chunkType_2) ? TRUE : FALSE;
        return typeMatch;
    }

    /**
     * Returns the given mention's governing VP chunk, in the given
     * caption; returns null if the mention has no governing VP or
     * if the caption has no root dependency node
     *
     * @param m
     * @param c
     * @return
     */
    private static Chunk getOwningVP(Mention m, Caption c) {
        DependencyNode rootNode = c.getRootNode();
        if (rootNode != null) {
            DependencyNode node = rootNode.findDependent(m.getHead());
            for (Integer chunkIdx : node.getGoverningChunkIndices()) {
                if (chunkIdx >= 0 && chunkIdx < c.getChunkList().size()) {
                    Chunk ch = c.getChunkList().get(chunkIdx);
                    if (ch.getChunkType().equals("VP"))
                        return ch;
                }
            }
        }
        return null;
    }


    public static void export_affinityFeats(Collection<Document> docSet, String dataSplit)
    {
        List<Document> docList = new ArrayList<>(docSet);

        Logger.log("Initializing stop words"); //stop words courtesy of http://www.ranks.nl/
        _stopWords = new HashSet<>(FileIO.readFile_lineList(Overlord.resourcesDir + "stop_words.txt"));

        Logger.log("Reading vocabulary from documents");
        Set<String> vocabulary = new HashSet<>();
        for (Document d : docList){
            for (Mention m : d.getMentionList()){
                for (Token t : m.getTokenList()) {
                    String text = t.toString().toLowerCase().trim();
                    if (!_stopWords.contains(text) && StringUtil.hasAlphaNum(text))
                        vocabulary.add(text);
                }
            }
        }

        Logger.log("Loading Word2Vec for vocabulary");
        _w2vUtil = new Word2VecUtil(Overlord.word2vecPath, vocabulary);

        Logger.log("Preprocessing documents");
        Set<String> boxFiles = new HashSet<>();
        File boxDir = new File(Overlord.boxFeatureDir);
        for(File f : boxDir.listFiles())
            if(f.isFile())
                boxFiles.add(f.getName().replace(".feats", ""));

        List<String> ll_affinity = new ArrayList<>();
        if(dataSplit.equals("train")){
            ll_affinity.addAll(export_affinityFeats_train(docSet, boxFiles));
        } else {
            int docIdx = 0;
            for(Document d : docList){
                List<String> ll_fvStr = FileIO.readFile_lineList(Overlord.boxFeatureDir + d.getID().replace(".jpg", ".feats"));
                Map<String, List<Double>> fvDict = new HashMap<>();
                for(String fvStr : ll_fvStr){
                    FeatureVector fv = FeatureVector.parseFeatureVector(fvStr);
                    List<Double> boxFeats = new ArrayList<>();
                    for (int i = 1; i <= 4096; i++)
                        boxFeats.add(fv.getFeatureValue(i));
                    fvDict.put(fv.comments, boxFeats);
                }
                for(Mention m : d.getMentionList()){
                    Set<BoundingBox> boxSet = d.getBoxSetForMention(m);
                    for(BoundingBox b : d.getBoundingBoxSet()){
                        int label = boxSet.contains(b) ? 1 : 0;
                        String ID = m.getUniqueID() + "|" + b.getUniqueID();
                        List<Double> feats_affinity = new ArrayList<>();
                        fvDict.get(b.getUniqueID()).stream().forEachOrdered(v -> feats_affinity.add(v));
                        _w2vUtil.getVector(m.toString().toLowerCase().trim()).stream().forEachOrdered(v -> feats_affinity.add(v));
                        ll_affinity.add(ID + "," + label + "," + StringUtil.listToString(feats_affinity, ","));
                    }
                }
                docIdx++;
                Logger.logStatus("complete (%.2f%%)", 100.0 * (double)docIdx /
                        docSet.size());
            }
        }

        FileIO.writeFile(ll_affinity, Overlord.dataPath +
                "feats/affinity_feats_" + dataSplit, "feats", false);
    }

    private static List<String> export_affinityFeats_train(Collection<Document> docSet, Set<String> boxFiles)
    {
        List<String> ll_affinity = new ArrayList<>();

        //Store a mapping of [docID -> [mention -> [bounding boxes] ] ]
        Map<String, Map<Mention, Set<BoundingBox>>>
                mentionBoxesDict = new HashMap<>();
        Map<Mention, Document> mentionDocDict = new HashMap<>();
        for(Document d : docSet){
            if(boxFiles.contains(d.getID().replace(".jpg", ""))){
                for(Mention m : d.getMentionList()){
                    mentionDocDict.put(m, d);
                    String normText = m.toString().toLowerCase().trim();
                    if(!mentionBoxesDict.containsKey(normText))
                        mentionBoxesDict.put(normText, new HashMap<>());
                    mentionBoxesDict.get(normText).put(m, d.getBoxSetForMention(m));
                }
            } else {
                System.out.println("ERROR: found no box feats for " + d.getID());
            }
        }

        //Randomly sample 10 bounding boxes from those that have boxes if
        //we're subsampling
        Map<String, Set<BoundingBox>> mentionBoxesDict_pos = new HashMap<>();
        Map<String, Set<BoundingBox>> mentionBoxesDict_neg = new HashMap<>();
        for(String normText : mentionBoxesDict.keySet()){
            List<BoundingBox> boxList_pos = new ArrayList<>();
            List<BoundingBox> boxList_neg = new ArrayList<>();
            for(Mention m : mentionBoxesDict.get(normText).keySet()){
                boxList_pos.addAll(mentionBoxesDict.get(normText).get(m));
                Document d = mentionDocDict.get(m);
                Set<BoundingBox> boxSet = new HashSet<>(d.getBoundingBoxSet());
                boxSet.removeAll(d.getBoxSetForMention(m));
                boxList_neg.addAll(boxSet);
            }
            if(!boxList_pos.isEmpty() && !boxList_neg.isEmpty()){
                Collections.shuffle(boxList_pos);
                Collections.shuffle(boxList_neg);
                mentionBoxesDict_pos.put(normText, new HashSet<>(boxList_pos.subList(0, Math.min(boxList_pos.size()-1, 10))));
                mentionBoxesDict_neg.put(normText, new HashSet<>(boxList_neg.subList(0, Math.min(boxList_neg.size()-1, 10))));
            }
        }

        int txtIdx = 0;
        for(String normText : mentionBoxesDict_pos.keySet()){
            List<Double> feats_txt = _w2vUtil.getVector(normText);

            //Since we don't know which image word's box will come from
            //a-priori, open the doc's file at each box (less efficient, but necessary)
            for(BoundingBox b : mentionBoxesDict_pos.get(normText)){
                List<Double> feats_img = null;
                try {
                    BufferedReader br = new BufferedReader(new InputStreamReader(
                            new FileInputStream(Overlord.boxFeatureDir +
                                    b.getDocID().replace(".jpg", ".feats"))));
                    String nextLine = br.readLine();
                    while (nextLine != null && feats_img == null) {
                        String fvID = nextLine.split(" # ")[1];
                        if (fvID.equals(b.getUniqueID())) {
                            FeatureVector fv = FeatureVector.parseFeatureVector(nextLine);
                            feats_img = new ArrayList<>();
                            for (int i = 1; i <= 4096; i++)
                                feats_img.add(fv.getFeatureValue(i));
                        }
                        nextLine = br.readLine();
                    }
                    br.close();
                    txtIdx++;
                    Logger.logStatus("text complete (%.2f%%)", 100.0 * (double)txtIdx /
                            (mentionBoxesDict_pos.keySet().size() * 2));
                } catch(Exception ex){Logger.log(ex);}

                if(feats_img == null){
                    Logger.log("ERROR: found no feats for " + b.getUniqueID());
                    continue;
                }

                List<Double> feats_affinity = new ArrayList<>();
                feats_img.stream().forEachOrdered(v -> feats_affinity.add(v));
                feats_txt.stream().forEachOrdered(v -> feats_affinity.add(v));
                String ID = "";
                ll_affinity.add(ID + ",1," + StringUtil.listToString(feats_affinity, ","));
            }
        }
        for(String normText : mentionBoxesDict_neg.keySet()){
            List<Double> feats_txt = _w2vUtil.getVector(normText);

            //Since we don't know which image word's box will come from
            //a-priori, open the doc's file at each box (less efficient, but necessary)
            for(BoundingBox b : mentionBoxesDict_pos.get(normText)){
                List<Double> feats_img = null;
                try {
                    BufferedReader br = new BufferedReader(new InputStreamReader(
                            new FileInputStream(Overlord.boxFeatureDir +
                                    b.getDocID().replace(".jpg", ".feats"))));
                    String nextLine = br.readLine();
                    while (nextLine != null && feats_img == null) {
                        String fvID = nextLine.split(" # ")[1];
                        if (fvID.equals(b.getUniqueID())) {
                            FeatureVector fv = FeatureVector.parseFeatureVector(nextLine);
                            feats_img = new ArrayList<>();
                            for (int i = 1; i <= 4096; i++)
                                feats_img.add(fv.getFeatureValue(i));
                        }
                        nextLine = br.readLine();
                    }
                    br.close();
                    txtIdx++;
                    Logger.logStatus("text complete (%.2f%%)", 100.0 * (double)txtIdx /
                            (mentionBoxesDict_neg.keySet().size() * 2));
                } catch(Exception ex){Logger.log(ex);}

                if(feats_img == null){
                    Logger.log("ERROR: found no feats for " + b.getUniqueID());
                    continue;
                }

                List<Double> feats_affinity = new ArrayList<>();
                feats_img.stream().forEachOrdered(v -> feats_affinity.add(v));
                feats_txt.stream().forEachOrdered(v -> feats_affinity.add(v));
                String ID = "";
                ll_affinity.add(ID + ",0," + StringUtil.listToString(feats_affinity, ","));
            }
        }
        return ll_affinity;
    }

    /**
     * Exports the joint box/mention embedding used in the affinity model, resulting
     * in a collection of [4096 + 300] vectors, for the img + txt feats
     *
     * @param docSet
     * @param outroot
     * @param numThreads
     */
    @Deprecated
    public static void exportFeatures_affinity(Collection<Document> docSet,
                                               String outroot, int numThreads)
    {
        _outroot = outroot;
        List<Document> docList = new ArrayList<>(docSet);
        //docList = docList.subList(0, 100);

        Logger.log("Initializing lists"); //stop words courtesy of http://www.ranks.nl/
        ClassifyUtil.initLists();

        Logger.log("Reading vocabulary from documents");
        Set<String> vocabulary = new HashSet<>();
        for (Document d : docList){
            for (Mention m : d.getMentionList()){
                for (Token t : m.getTokenList()) {
                    String text = t.toString().toLowerCase().trim();
                    if (!_stopWords.contains(text) && StringUtil.hasAlphaNum(text))
                        vocabulary.add(text);
                }
            }
        }
        Logger.log("Loading Word2Vec for vocabulary");
        _w2vUtil = new Word2VecUtil(Overlord.word2vecPath, vocabulary);

        Logger.log("Iterating through documents, separately loading box features for each");
        ExtractionThread[] threadPool = new ExtractionThread[numThreads];
        //create and start our first set of threads
        int docIdx = 0;
        for(int i=0; i<numThreads; i++) {
            Document d = docList.get(docIdx);
            threadPool[i] = new ExtractionThread(d, ExtractionThreadType.AFFINITY);
            docIdx++;
        }

        for(int i=0; i<threadPool.length; i++)
            threadPool[i].start();
        boolean foundLiveThread = true;
        while(docIdx < docList.size() || foundLiveThread) {
            foundLiveThread = false;
            for(int i=0; i<numThreads; i++) {
                if(threadPool[i].isAlive()) {
                    foundLiveThread = true;
                } else {
                    //independently, if we found a dead thread and we
                    //still have image IDs to iterate through, swap this
                    //dead one out for a live one
                    if(docIdx < docList.size()) {
                        Logger.logStatus("Processed %d docs (%.2f%%)",
                                docIdx, 100.0*(double)docIdx / docList.size());
                        Document d = docList.get(docIdx);
                        threadPool[i] = new ExtractionThread(d, ExtractionThreadType.AFFINITY);
                        docIdx++;
                        threadPool[i].start();
                        foundLiveThread = true;
                    }
                }
            }

            //before we check for threadlife again, let's
            //sleep 50ms so we don't burn
            try{Thread.sleep(50);}
            catch(InterruptedException iEx){/*do nothing*/}
        }
        Logger.log("Done");
    }

    private static void exportFeatures(Collection<Document> docSet,
                                       ExtractionThreadType featureType,
                                       int numThreads)
    {
        Logger.log("Opening [" +_outroot+ "] for writing");
        BufferedWriter bw = null;
        try {
            bw = new BufferedWriter(new FileWriter(_outroot));
        } catch(IOException ioEx) {
            System.err.println("Could not save output file " + _outroot);
            System.exit(0);
        }

        Logger.log("Computing " + featureType.toString() + " classifier features");
        List<Document> docList = new ArrayList<>(docSet);
        ExtractionThread[] threadPool = new ExtractionThread[numThreads];

        //create and start our first set of threads
        int docIdx = 0;
        for(int i=0; i<numThreads; i++) {
            Document d = docList.get(docIdx);
            threadPool[i] = new ExtractionThread(d, featureType);
            docIdx++;
        }

        for(int i=0; i<threadPool.length; i++)
            threadPool[i].start();
        boolean foundLiveThread = true;
        Set<String> addedDocIDs = new HashSet<>();
        DoubleDict<Integer> labelDistro = new DoubleDict<>();
        while(docIdx < docList.size() || foundLiveThread) {
            foundLiveThread = false;
            for(int i=0; i<numThreads; i++) {
                if(threadPool[i].isAlive()) {
                    foundLiveThread = true;
                } else {
                    addedDocIDs.add(threadPool[i]._doc.getID());
                    //if this is a dead thread, store it
                    try{
                        for(FeatureVector fv : threadPool[i].fvSet) {
                            bw.write(fv.toString() + "\n");
                            labelDistro.increment((int)fv.label);
                        }
                        addedDocIDs.add(threadPool[i]._doc.getID());
                    } catch (IOException ioEx){
                        Logger.log(ioEx);
                    }

                    //independently, if we found a dead thread and we
                    //still have image IDs to iterate through, swap this
                    //dead one out for a live one
                    if(docIdx < docList.size()) {
                        Logger.logStatus("Processed %d images (%.2f%%)",
                                docIdx, 100.0*(double)docIdx / docList.size());
                        Document d = docList.get(docIdx);
                        threadPool[i] = new ExtractionThread(d, featureType);
                        docIdx++;
                        threadPool[i].start();
                        foundLiveThread = true;
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
            if(!addedDocIDs.contains(threadPool[i]._doc.getID())){
                try{
                    for(FeatureVector fv : threadPool[i].fvSet) {
                        bw.write(fv.toString() + "\n");
                        labelDistro.increment((int)fv.label);
                    }
                    addedDocIDs.add(threadPool[i]._doc.getID());
                } catch (IOException ioEx){
                    Logger.log(ioEx);
                }
                addedDocIDs.add(threadPool[i]._doc.getID());
            }
        }

        Logger.log("Label distro");
        System.out.println(labelDistro.toString());

        Logger.log("Closing [" + _outroot + "]");
        try {
            bw.close();
        } catch(IOException ioEx) {
            System.err.println("Could not save output file " + _outroot);
        }
    }

    /**Returns an alphabetized list of strings in the given filename
     * that occur more than thresh times; originally implemented to
     * help produce one-hot vectors
     *
     * @param filename
     * @param thresh
     * @return
     */
    private static List<String> getOneHotList(String filename, int thresh)
    {
        List<String> strList = new ArrayList<>();
        String[][] table = FileIO.readFile_table(filename);
        for(String[] row : table)
            if(Double.parseDouble(row[1]) > thresh)
                strList.add(row[0]);
        Collections.sort(strList);
        return strList;
    }

    public static boolean getGenderMatch(Mention m1, Mention m2)
    {
        String gender_1 = m1.getGender();
        String gender_2 = m2.getGender();
        return gender_1.equals(gender_2) || gender_1.equals("neuter") ||
                gender_2.equals("neuter");
    }

    /**Returns whether the lexical types for m1 and m2 intersect
     *
     *
     * @param m1
     * @param m2
     * @return
     */
    public static boolean getTypeMatch(Mention m1, Mention m2)
    {
        Set<String> typeSet_1 =
                new HashSet<>(Arrays.asList(m1.getLexicalType().split("/")));
        Set<String> typeSet_2 =
                new HashSet<>(Arrays.asList(m2.getLexicalType().split("/")));
        Set<String> intersection = new HashSet<>(typeSet_1);
        intersection.retainAll(typeSet_2);
        return intersection.size() > 0;
    }

    private enum ExtractionThreadType
    {
        AFFINITY, PAIRWISE
    }

    private static class ExtractionThread extends Thread
    {
        private static Map<String, Object> _metaDict = new HashMap<>();

        private Document _doc;
        private ExtractionThreadType _type;
        private Set<String> _subsetMentions;
        public Collection<FeatureVector> fvSet;

        public ExtractionThread(Document doc, ExtractionThreadType type)
        {
            _doc = doc;
            _type = type;
            fvSet = new HashSet<>();
            _subsetMentions = _doc.getSubsetMentions();
        }

        public void run()
        {
            switch(_type){
                case AFFINITY: run_affinity();
                    break;
                case PAIRWISE: run_pairwise();
                    break;
            }
        }

        private void run_affinity()
        {
            //read our box features for this image
            List<String> ll_boxFeats =
                    FileIO.readFile_lineList(Overlord.boxFeatureDir +
                            _doc.getID().replace(".jpg", ".feats"));
            Map<String, FeatureVector> boxFeats = new HashMap<>();
            for(String featStr : ll_boxFeats){
                FeatureVector fv = FeatureVector.parseFeatureVector(featStr);
                boxFeats.put(fv.comments, fv);
            }
            if(ll_boxFeats.isEmpty())
                Logger.log("Found no box features for " + _doc.getID());

            //load our mention features for all mentions in this doc
            Map<String, List<Double>> mentionFeats = new HashMap<>();
            for(Mention m : _doc.getMentionList()){
                List<List<Double>> vectorList = new ArrayList<>();
                for(Token t : m.getTokenList()){
                    List<Double> vecToAdd = _w2vUtil.getVector(t.toString().toLowerCase().trim());
                    if(vecToAdd.size() < 300){
                        System.err.printf("Token %s has vector size %d\n",
                                t.toString(), vecToAdd.size());
                    }
                    vectorList.add(vecToAdd);
                }
                List<Double> meanVec = Util.vectorMean(vectorList);
                if(meanVec.size() < 300){
                    System.err.printf("Mention %s has vector size %d\n",
                            m.toString(), meanVec.size());
                }
                mentionFeats.put(m.getUniqueID(), meanVec);
            }

            //finally, concatenate and save the vectors
            List<String> vectors = new ArrayList<>();
            List<String> vectors_pos = new ArrayList<>();
            List<String> vectors_neg = new ArrayList<>();
            Set<String> posWords = new HashSet<>();
            for(BoundingBox b : _doc.getBoundingBoxSet()){
                FeatureVector fv_box = boxFeats.get(b.getUniqueID());
                Set<Mention> mentionSet_box = _doc.getMentionSetForBox(b);

                for(Mention m : _doc.getMentionList()){
                    List<Double> fv_mention = mentionFeats.get(m.getUniqueID());

                    List<Double> concat = new ArrayList<>();
                    for(int i=1; i<=4096; i++) {
                        Double val = 0.0;
                        if (fv_box != null) {
                            Double x = fv_box.getFeatureValue(i);
                            if (x != null)
                                val = x;
                        }
                        concat.add(val);
                    }
                    fv_mention.stream().forEachOrdered(v -> concat.add(v));

                    int label = 0;
                    if(mentionSet_box.contains(m))
                        label = 1;

                    String ID = b.getUniqueID() + "|" + m.getUniqueID();

                    /*
                    if(label == 1) {
                        posWords.add(m.getHead().toString().toLowerCase().trim());
                        vectors_pos.add(ID + "," + label + "," + StringUtil.listToString(concat, ","));
                    } else if(!posWords.contains(m.getHead().toString().toLowerCase().trim())){
                        vectors_neg.add(ID + "," + label + "," + StringUtil.listToString(concat, ","));
                    }*/
                    vectors.add(ID + "," + label + "," + StringUtil.listToString(concat, ","));
                }
            }
            //List<String> vectors = new ArrayList<>();
            //vectors.addAll(vectors_pos);
            //Collections.shuffle(vectors_neg);
            //vectors.addAll(vectors_neg.subList(0, Math.min(vectors_pos.size(), vectors_neg.size())-1));
            FileIO.writeFile(vectors, _outroot + _doc.getID().replace(".jpg", ""), "feats", false);
        }

        public static Map<String, Object> getMetaDict()
        {
            return _metaDict;
        }

        private void run_pairwise()
        {
            List<Mention> mentionList = _doc.getMentionList();
            for(int i=0; i<mentionList.size(); i++){
                Mention m_i = mentionList.get(i);
                Mention.PRONOUN_TYPE type_i = m_i.getPronounType();
                if(type_i == Mention.PRONOUN_TYPE.NONE || type_i == Mention.PRONOUN_TYPE.SEMI){
                    //if this is a training document, don't bother training with
                    //nonvisual mentions
                    if(_doc.getIsTrain() && m_i.getChainID().equals("0"))
                        continue;

                    for(int j=i+1; j<mentionList.size(); j++){
                        Mention m_j = mentionList.get(j);
                        Mention.PRONOUN_TYPE type_j = m_j.getPronounType();
                        if(type_j == Mention.PRONOUN_TYPE.NONE || type_j == Mention.PRONOUN_TYPE.SEMI){
                            if(_doc.getIsTrain() && m_j.getChainID().equals("0"))
                                continue;

                            fvSet.add(getPairwiseFeatureVector(m_i, m_j));
                            fvSet.add(getPairwiseFeatureVector(m_j, m_i));
                        }
                    }
                }
            }
        }

        /**Return a complete pairwise feature vector
         *
         * @param m1
         * @param m2
         * @return
         */
        private FeatureVector getPairwiseFeatureVector(Mention m1, Mention m2)
        {
            int currentIdx = 1;
            List<Object> featureList = new ArrayList<>();

            //Head matches
            String head_1 = m1.getHead().toString().toLowerCase();
            String head_2 = m2.getHead().toString().toLowerCase();
            Integer f_headMatch = head_1.equals(head_2) ? TRUE : FALSE;
            Integer f_headPOSMatch = m1.getHead().getPosTag().equals(m2.getHead().getPosTag()) ? TRUE : FALSE;
            featureList.add(f_headMatch);
            addMetaEntry("head_match", currentIdx++);
            featureList.add(f_headPOSMatch);
            addMetaEntry("head_pos_match", currentIdx++);

            //Lemma match / substring feat
            String lemma_1 = m1.getHead().getLemma().toLowerCase();
            String lemma_2 = m2.getHead().getLemma().toLowerCase();
            Integer f_lemmaMatch = lemma_1.equals(lemma_2) ? TRUE : FALSE;
            Integer f_substring = lemma_1.contains(lemma_2) ||
                    lemma_2.contains(lemma_1) ? TRUE : FALSE;
            featureList.add(f_lemmaMatch);
            addMetaEntry("lemma_match", currentIdx++);
            featureList.add(f_substring);
            addMetaEntry("substring_match", currentIdx++);

            //Extent match
            String extent_1 = m1.toString().replace(m1.getHead().toString(), "").toLowerCase();
            String extent_2 = m2.toString().replace(m2.getHead().toString(), "").toLowerCase();
            Integer f_extentMatch = UNK;
            if(!extent_1.isEmpty() || !extent_2.isEmpty())
                f_extentMatch = extent_1.equalsIgnoreCase(extent_2) ? TRUE : FALSE;
            featureList.add(f_extentMatch);
            addMetaEntry("extent_match", currentIdx++);

            //Type match
            String type_1 = m1.getLexicalType();
            String type_2 = m2.getLexicalType();
            Double f_lexTypeMatch = (double)UNK;
            Double f_lexTypeMatch_other = (double)UNK;
            Integer f_lexTypeMatch_only = UNK;
            if(type_1 != null && type_2 != null) {
                f_lexTypeMatch = Mention.getLexicalTypeMatch(m1, m2);
                if(f_lexTypeMatch == 0.0)
                    f_lexTypeMatch = (double)FALSE;

                //If both are strictly other, 1; if both _contain_ other, 0.5; else -1
                if(type_1.equals("other") && type_2.equals("other"))
                    f_lexTypeMatch_other = (double)TRUE;
                else if(type_1.contains("other") && type_2.contains("other"))
                    f_lexTypeMatch_other = 0.5;
                else
                    f_lexTypeMatch_other = (double)FALSE;

                if(f_lexTypeMatch == 1.0 &&
                   _onlyTypeMentions.contains(m1) &&
                   _onlyTypeMentions.contains(m2)) {
                    f_lexTypeMatch_only = TRUE;
                } else {
                    f_lexTypeMatch_only = FALSE;
                }
            }
            featureList.add(f_lexTypeMatch);
            addMetaEntry("lex_type_match", currentIdx++);
            featureList.add(f_lexTypeMatch_other);
            addMetaEntry("lex_type_match_other", currentIdx++);
            featureList.add(f_lexTypeMatch_only);
            addMetaEntry("lex_type_match_only", currentIdx++);

            //Chunk neighbor features -- left
            Chunk leftNeighbor_1 = _mentionChunkNeighborDict.get(m1)[0];
            Chunk leftNeighbor_2 = _mentionChunkNeighborDict.get(m2)[0];
            String leftChunkType_1 = null, leftChunkType_2 = null;
            if(leftNeighbor_1 != null)
                leftChunkType_1 = leftNeighbor_1.getChunkType();
            if(leftNeighbor_2 != null)
                leftChunkType_2 = leftNeighbor_2.getChunkType();
            Integer f_leftMatch = getChunkTypeMatch(leftChunkType_1, leftChunkType_2);
            featureList.add(f_leftMatch);
            addMetaEntry("left_chunk_match", currentIdx++);

            //Chunk neighbor features -- right
            Chunk rightNeighbor_1 = _mentionChunkNeighborDict.get(m1)[1];
            Chunk rightNeighbor_2 = _mentionChunkNeighborDict.get(m2)[1];
            String rightChunkType_1 = null, rightChunkType_2 = null;
            if(rightNeighbor_1 != null)
                rightChunkType_1 = rightNeighbor_1.getChunkType();
            if(rightNeighbor_2  != null)
                rightChunkType_2 = rightNeighbor_2.getChunkType();
            Integer f_rightMatch = getChunkTypeMatch(rightChunkType_1, rightChunkType_2);
            featureList.add(f_rightMatch);
            addMetaEntry("right_chunk_match", currentIdx++);

            //Dependency tree features
            DependencyNode root_1 = _doc.getCaption(m1.getCaptionIdx()).getRootNode();
            DependencyNode root_2 = _doc.getCaption(m2.getCaptionIdx()).getRootNode();
            Integer f_outDepMatch = UNK;
            if(root_1 != null && root_2 != null){
                Set<String> outRel_1 = root_1.getOutRelations(m1);
                Set<String> outRel_2 = root_2.getOutRelations(m2);
                Set<String> outRel = new HashSet<>(outRel_1);
                outRel.retainAll(outRel_2);
                f_outDepMatch = outRel.isEmpty() ? FALSE : TRUE;
            }
            featureList.add(f_outDepMatch);
            addMetaEntry("out_dep_match", currentIdx++);

            //Determiner plural match (assume the first word is
            //the determiner candidate); FALSE is only assigned when
            //both have determiners of different pluralities
            String firstWord_1 = m1.getTokenList().get(0).toString().toLowerCase();
            String firstWord_2 = m2.getTokenList().get(0).toString().toLowerCase();
            Integer f_detPluralMatch = UNK;
            if(_detSet_singular.contains(firstWord_1) && _detSet_singular.contains(firstWord_2))
                f_detPluralMatch = TRUE;
            else if(_detSet_plural.contains(firstWord_1) && _detSet_plural.contains(firstWord_2))
                f_detPluralMatch = TRUE;
            else if(_detSet_singular.contains(firstWord_1) && _detSet_plural.contains(firstWord_2))
                f_detPluralMatch = FALSE;
            else if(_detSet_plural.contains(firstWord_1) && _detSet_singular.contains(firstWord_2))
                f_detPluralMatch = FALSE;
            featureList.add(f_detPluralMatch);
            addMetaEntry("det_plural_match", currentIdx++);

            //Verb features
            Chunk subjOf_1 = _subjOfDict.get(m1), subjOf_2 = _subjOfDict.get(m2);
            Chunk objOf_1 = _objOfDict.get(m1), objOf_2 = _objOfDict.get(m2);
            String subjOfStr_1 = subjOf_1 == null ? null :
                    subjOf_1.getTokenList().get(subjOf_1.getTokenList().size()-1).toString().toLowerCase();
            String subjOfStr_2 = subjOf_2 == null ? null :
                    subjOf_2.getTokenList().get(subjOf_2.getTokenList().size()-1).toString().toLowerCase();
            String objOfStr_1 = objOf_1 == null ? null :
                    objOf_1.getTokenList().get(objOf_1.getTokenList().size()-1).toString().toLowerCase();
            String objOfStr_2 = objOf_2 == null ? null :
                    objOf_2.getTokenList().get(objOf_2.getTokenList().size()-1).toString().toLowerCase();
            //whether each individual position is a subject / object
            Integer f_isSubj_1 = subjOf_1 != null ? TRUE : FALSE;
            Integer f_isSubj_2 = subjOf_2 != null ? TRUE : FALSE;
            Integer f_isObj_1 = objOf_1 != null ? TRUE : FALSE;
            Integer f_isObj_2 = objOf_2 != null ? TRUE : FALSE;
            //whether both mentions are subjects; both mentions are objects
            Integer f_isSubjMatch = f_isSubj_1 == TRUE && f_isSubj_2 == TRUE ? TRUE : FALSE;
            Integer f_isObjMatch = f_isObj_1 == TRUE && f_isObj_2 == TRUE ? TRUE : FALSE;

            //whether the subjects and objects match
            Integer f_subjOfMatch = UNK;
            if(subjOfStr_1 != null)
                f_subjOfMatch = subjOfStr_1.equals(subjOfStr_2) ? TRUE : FALSE;
            else if(subjOfStr_2 != null) //reaching here is always a non-match
                f_subjOfMatch = FALSE;
            Integer f_objOfMatch = UNK;
            if(objOfStr_1 != null)
                f_objOfMatch = objOfStr_1.equals(objOfStr_2) ? TRUE : FALSE;
            else if(objOfStr_2 != null) //reaching here is always a non-match
                f_objOfMatch = FALSE;
            featureList.add(f_isSubjMatch);
            featureList.add(f_isObjMatch);
            addMetaEntry("is_subj_match", currentIdx++);
            addMetaEntry("is_obj_match", currentIdx++);
            featureList.add(f_subjOfMatch);
            featureList.add(f_objOfMatch);
            addMetaEntry("subj_of_match", currentIdx++);
            addMetaEntry("obj_of_match", currentIdx++);
            featureList.add(f_isSubj_1);
            featureList.add(f_isSubj_2);
            addMetaEntry("is_subj_1", currentIdx++);
            addMetaEntry("is_subj_2", currentIdx++);
            featureList.add(f_isObj_1);
            featureList.add(f_isObj_2);
            addMetaEntry("is_obj_1", currentIdx++);
            addMetaEntry("is_obj_2", currentIdx++);

            //features for semi-pronouns
            Integer f_semiPronom_1 = m1.getPronounType() == Mention.PRONOUN_TYPE.SEMI ? TRUE : FALSE;
            Integer f_semiPronom_2 = m2.getPronounType() == Mention.PRONOUN_TYPE.SEMI ? TRUE : FALSE;
            Caption cap1 = _doc.getCaption(m1.getCaptionIdx());
            Caption cap2 = _doc.getCaption(m2.getCaptionIdx());
            Integer f_xOfY_1 = FALSE, f_xOfY_2 = FALSE;
            if(m1.getIdx() + 1 < cap1.getMentionList().size()){
                List<Token> intrstlTokens =
                        cap1.getInterstitialTokens(m1, cap1.getMentionList().get(m1.getIdx() + 1));
                if(intrstlTokens.size() == 1 && intrstlTokens.get(0).toString().equals("of"))
                    f_xOfY_1 = TRUE;
            }
            if(m2.getIdx() + 1 < cap2.getMentionList().size()){
                List<Token> intrstlTokens =
                        cap2.getInterstitialTokens(m2, cap2.getMentionList().get(m2.getIdx() + 1));
                if(intrstlTokens.size() == 1 && intrstlTokens.get(0).toString().equals("of"))
                    f_xOfY_2 = TRUE;
            }
            Integer f_appos_1 = FALSE, f_appos_2 = FALSE;
            if(cap1.toString().toLowerCase().matches(PTRN_APPOS)){
                //Now that we know that this caption has an appositive
                //construction, this mention is in one when it
                //isn't the first mention and appears before the second comma
                if(m1.getIdx() > 0){
                    Token secondComma = null;
                    boolean seenOneComma = false;
                    for(Token t : cap1.getTokenList()){
                        if(t.toString().equals(",")){
                            if(seenOneComma) {
                                secondComma = t;
                                break;
                            } else {
                                seenOneComma = true;
                            }
                        }
                    }
                    if(secondComma != null && m1.getTokenRange()[1] < secondComma.getIdx())
                        f_appos_1 = TRUE;
                }
            }
            if(cap2.toString().toLowerCase().matches(PTRN_APPOS)){
                if(m2.getIdx() > 0){
                    Token secondComma = null;
                    boolean seenOneComma = false;
                    for(Token t : cap2.getTokenList()){
                        if(t.toString().equals(",")){
                            if(seenOneComma) {
                                secondComma = t;
                                break;
                            } else {
                                seenOneComma = true;
                            }
                        }
                    }
                    if(secondComma != null && m2.getTokenRange()[1] < secondComma.getIdx())
                        f_appos_2 = TRUE;
                }
            }
            Integer f_inList_1 = FALSE, f_inList_2 = FALSE;
            if(cap1.toString().toLowerCase().matches(PTRN_LIST)){
                //Now that we know the caption contains a list among
                //the first mentions, find the first 'and' to detect
                //the mentions around it
                Token firstAnd = null;
                for(Token t : cap1.getTokenList()){
                    if(t.toString().equals("and")){
                        firstAnd = t;
                        break;
                    }
                }
                if(firstAnd != null &&
                  (m1.getTokenRange()[1] < firstAnd.getIdx() ||
                   firstAnd.getIdx() + 1 == m1.getTokenRange()[0])){
                    f_inList_1 = TRUE;
                }
            }
            if(cap2.toString().toLowerCase().matches(PTRN_LIST)){
                Token firstAnd = null;
                for(Token t : cap2.getTokenList()){
                    if(t.toString().equals("and")){
                        firstAnd = t;
                        break;
                    }
                }
                if(firstAnd != null &&
                  (m2.getTokenRange()[1] < firstAnd.getIdx() ||
                  firstAnd.getIdx() + 1 == m2.getTokenRange()[0])){
                    f_inList_2 = TRUE;
                }
            }
            featureList.add(f_semiPronom_1);
            featureList.add(f_semiPronom_2);
            addMetaEntry("semi_pronom_1", currentIdx++);
            addMetaEntry("semi_pronom_2", currentIdx++);
            featureList.add(f_xOfY_1);
            featureList.add(f_xOfY_2);
            addMetaEntry("x_of_y_1", currentIdx++);
            addMetaEntry("x_of_y_2", currentIdx++);
            featureList.add(f_appos_1);
            featureList.add(f_appos_2);
            addMetaEntry("appositive_1", currentIdx++);
            addMetaEntry("appositive_2", currentIdx++);
            featureList.add(f_inList_1);
            featureList.add(f_inList_2);
            addMetaEntry("in_list_1", currentIdx++);
            addMetaEntry("in_list_2", currentIdx++);

            //Meta features
            Integer f_headNotLemma = f_headMatch == TRUE && f_lemmaMatch == FALSE ? TRUE : FALSE;
            Integer f_lemmaNotHead = f_lemmaMatch == TRUE && f_headMatch == FALSE ? TRUE : FALSE;
            featureList.add(f_headNotLemma);
            featureList.add(f_lemmaNotHead);
            addMetaEntry("head_not_lemma", currentIdx++);
            addMetaEntry("lemma_not_head", currentIdx++);

            //Add all features to the vector
            FeatureVector fv = new FeatureVector();
            for(int k=0; k<featureList.size(); k++) {
                Double val = Double.parseDouble(featureList.get(k).toString());
                if(val != 0)
                    fv.addFeature(k+1, val);
            }

            //Get the pair strings for onehot vectors
            String headPair = head_1 + "|" + head_2;
            headPair = headPair.replace(",", "");
            String lemmaPair = lemma_1 + "|" + lemma_2;
            lemmaPair = lemmaPair.replace(",", "");
            String typePair = null;
            if(type_1 != null && type_2 != null)
                typePair = type_1 + "|" + type_2;
            String leftPair = "";
            if(leftNeighbor_1 != null && leftNeighbor_2 != null)
                leftPair = String.format("%s|%s", leftNeighbor_1.toString(), leftNeighbor_2.toString()).toLowerCase();
            String rightPair = "";
            if(rightNeighbor_1 != null && rightNeighbor_2 != null)
                rightPair = String.format("%s|%s", rightNeighbor_1.toString(), rightNeighbor_2.toString()).toLowerCase();
            String subjOfPair = subjOfStr_1 + "|" + subjOfStr_2;
            String objOfPair = objOfStr_1 + "|" + objOfStr_2;
            String[] mods_1 = m1.getModifiers(), mods_2 = m2.getModifiers();
            String numericPair = StringUtil.getAlphabetizedPair(mods_1[0], mods_2[0]);
            String modPair = mods_1[0] + "|" + mods_2[1];

            String left_prep_1 = _prepDict_left.get(m1);
            String left_prep_2 = _prepDict_left.get(m2);
            String leftPrepPair = "";
            if(left_prep_1 != null && left_prep_2 != null)
                leftPrepPair = left_prep_1 + "|" + left_prep_2;
            String right_prep_1 = _prepDict_right.get(m1);
            String right_prep_2 = _prepDict_right.get(m2);
            String rightPrepPair = "";
            if(right_prep_1 != null && right_prep_2 != null)
                rightPrepPair = right_prep_1 + "|" + right_prep_2;

            //Add one hot vectors, which internally adjust the feature vector but
            //doesn't adjust the idx
            int start = currentIdx, end = start + _headPairList.size() + 1;
            addOneHotVector(fv, start, _headPairList, headPair);
            addMetaEntry("head_pair_onehot", start, end);
            start = end; end = start + _lemmaPairList.size() + 1;
            addOneHotVector(fv, start, _lemmaPairList, lemmaPair);
            addMetaEntry("lemma_pair_onehot", start, end);
            start = end; end = start + _typePairList.size() + 1;
            addOneHotVector(fv, start, _typePairList, typePair);
            addMetaEntry("type_pair_onehot", start, end);
            start = end; end = start + _leftPairList.size() + 1;
            addOneHotVector(fv, start, _leftPairList, leftPair);
            addMetaEntry("left_pair_onehot", start, end);
            start = end; end = start + _rightPairList.size() + 1;
            addOneHotVector(fv, start, _rightPairList, rightPair);
            addMetaEntry("right_pair_onehot", start, end);
            start = end; end = start + _subjOfList.size() + 1;
            addOneHotVector(fv, start, _subjOfList, subjOfPair);
            addMetaEntry("subj_of_onehot", start, end);
            start = end; end = start + _objOfList.size() + 1;
            addOneHotVector(fv, start, _objOfList, objOfPair);
            addMetaEntry("obj_of_onehot", start, end);
            start = end; end = start + _detList.size() + 1;
            addOneHotVector(fv, start, _detList, firstWord_1);
            addMetaEntry("det_1_onehot", start, end);
            start = end; end = start + _detList.size() + 1;
            addOneHotVector(fv, start, _detList, firstWord_2);
            addMetaEntry("det_2_onehot", start, end);
            start = end; end = start + _numericPairs.size() + 1;
            addOneHotVector(fv, start, _numericPairs, numericPair);
            addMetaEntry("numeric_pair_onehot", start, end);
            start = end; end = start + _modifierPairs.size() + 1;
            addOneHotVector(fv, start, _modifierPairs, modPair);
            addMetaEntry("modifier_pair_onehot", start, end);
            start = end; end = start + _prepositionPairList.size() + 1;
            addOneHotVector(fv, start, _prepositionPairList, leftPrepPair);
            addMetaEntry("left_preposition_pair_onehot", start, end);
            start = end; end = start + _prepositionPairList.size() + 1;
            addOneHotVector(fv, start, _prepositionPairList, rightPrepPair);
            addMetaEntry("right_preposition_pair_onehot", start, end);

            //We treat hypernyms as -- not a onehot -- but a bag-of-words;
            //Given 24 core concepts, we keep a vector where entry ij
            //is 1 if one of the mentions has concept i in its senses hypernyms
            //and the other has concept j
            start = end; end = start + _hypernyms.size() + 1;
            Set<String> hypSet_1 = _hypDict.get(lemma_1);
            Set<String> hypSet_2 = _hypDict.get(lemma_2);
            for(int i=0; i<_hypernyms.size(); i++){
                String hyp_i = _hypernyms.get(i);
                for(int j=i; j<_hypernyms.size(); j++){
                    String hyp_j = _hypernyms.get(j);

                    if(hypSet_1.contains(hyp_i) && hypSet_2.contains(hyp_j) ||
                       hypSet_1.contains(hyp_j) && hypSet_2.contains(hyp_i))
                        fv.addFeature(currentIdx, 1.0);
                    currentIdx++;
                }
            }
            addMetaEntry("hypernym_bow", start, end);

            //Finally, add a three-way label indicating if these mentions are
            //coreferent, subset, or null\
            Integer label = 0;
            if(!m1.getChainID().equals("0") && !m2.getChainID().equals("0")){
               String id_ij = Document.getMentionPairStr(m1, m2, true, true);
               String id_ji = Document.getMentionPairStr(m2, m1, true, true);
                if(m1.getChainID().equals(m2.getChainID()))
                    label = 1;
                else if(_subsetMentions.contains(id_ij))
                    label = 2;  //subset relation; assumes other
                                //vector from this pair will be 3
                else if(_subsetMentions.contains(id_ji))
                    label = 3; //Superset relation; assumes other
                               //vector from this pair will be 2
            }

            addMetaEntry("max_idx", end+1);
            fv.label = label;
            fv.comments = Document.getMentionPairStr(m1, m2, true, true);
            return fv;
        }

        private void addOneHotVector(FeatureVector fv, int currentIdx,
                                    List<String> oneHotList, String item)
        {
            if(oneHotList.indexOf(item) > -1)
                fv.addFeature(currentIdx + oneHotList.indexOf(item) + 1, 1.0);
        }

        private static void addMetaEntry(String key, int idx)
        {
            if(!_metaDict.containsKey(key))
                _metaDict.put(key, idx);
        }

        private static void addMetaEntry(String key, int start, int end)
        {
            if(!_metaDict.containsKey(key))
                _metaDict.put(key, new Integer[]{start, end});
        }
    }
}
