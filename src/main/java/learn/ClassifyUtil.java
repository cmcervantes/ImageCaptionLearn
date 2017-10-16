package learn;

import core.Overlord;
import nlptools.StanfordAnnotator;
import nlptools.Word2VecUtil;
import nlptools.WordnetUtil;
import org.apache.commons.lang.ArrayUtils;
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
    protected static Map<Mention, String> _prepDict_left;
    protected static Map<Mention, String> _prepDict_right;

    //lists from files
    private static Set<String> _colors;
    protected static Set<String> _detSet_singular;
    protected static Set<String> _detSet_plural;
    protected static Set<String> _stopWords;
    protected static List<String> _hypernyms;

    //other static lists
    private static Set<String> _articles;
    private static Set<String> _prps;
    private static Set<String> _masses;
    private static Set<String> _collectives;
    private static Set<String> _portions;
    private static Map<String, Integer> _collectives_kv;
    private static Map<String, Integer> _quantifiers_kv;

    //onehot lists
    private static Map<String, Integer> _typePairs;
    private static Map<String, Integer> _leftPairs;
    private static Map<String, Integer> _rightPairs;
    private static Map<String, Integer> _headPairs;
    private static Map<String, Integer> _lemmaPairs;
    private static Map<String, Integer> _subjOfPairs;
    private static Map<String, Integer> _objOfPairs;
    private static Map<String, Integer> _modifierPairs;
    private static Map<String, Integer> _numericPairs;
    private static Map<String, Integer> _prepositionPairs;
    private static Map<String, Integer> _categories;
    private static Map<String, Integer> _categoryPairs;
    private static Map<String, Integer> _heads;
    private static Map<String, Integer> _modifiers;
    private static Map<String, Integer> _numerics;
    private static Map<String, Integer> _prepositions;
    private static Map<String, Integer> _types;
    private static Map<String, Integer> _lefts;
    private static Map<String, Integer> _rights;
    private static Map<String, Integer> _subjOfs;
    private static Map<String, Integer> _objOfs;
    private static Map<String, Integer> _pronouns;
    private static Map<String, Integer> _pronounTypes;
    private static Map<String, Integer> _dets;
    private static Map<String, Integer> _nonvisuals;
    private static Map<String, Integer> _distances;


    protected static final String pattern_aside = "^NP , (NP (VP |ADJP |PP |and )*)+,.*$";

    /**Loads a onehot index dictionary given a histogram file and a frequency cutoff;
     * specifying 1, for example, loads the dictionary with all entries which have frequency
     * greater than 1.
     * NOTE: This stores indices as a dictionary, rather than simply as a list, for efficiency
     * concerns. The additional memory overhead of a hashmap is worth the ~35x speedup gained by
     * constant time lookups of onehot indices
     *
     * @param filename
     * @param freqCutoff
     * @return
     */
    static Map<String, Integer> loadOnehotDict(String filename, int freqCutoff)
    {
        Map<String, Integer> idxDict = new HashMap<>();
        String[][] table = FileIO.readFile_table(filename);
        int idx = 0;
        for(int i=0; i<table.length; i++){
            String[] row = table[i];
            if(Double.parseDouble(row[1]) > freqCutoff)
                idxDict.put(row[0], idx++); //idx not i because we don't want to count
                                            //entries below the cutoff
        }
        return idxDict;
    }

    /**Feature preprocessing loads onehot dictionaries and lists from files
     *
     * @param docSet
     */
    private static void _featurePreprocessing(Collection<Document> docSet)
    {
        Mention.initializeLexicons(Overlord.flickr30k_lexicon, Overlord.mscoco_lexicon);

        Logger.log("Feature preprocessing (onehot index dictionaries)");
        _typePairs = loadOnehotDict(Overlord.flickr30kResources + "hist_typePair_ordered.csv", 1000);
        _leftPairs = loadOnehotDict(Overlord.flickr30kResources + "hist_leftPair_ordered.csv", 1000);
        _rightPairs = loadOnehotDict(Overlord.flickr30kResources + "hist_rightPair_ordered.csv", 1000);
        _headPairs = loadOnehotDict(Overlord.flickr30kResources + "hist_headPair_ordered.csv", 1);
        _lemmaPairs = loadOnehotDict(Overlord.flickr30kResources + "hist_lemmaPair_ordered.csv", 1);
        _subjOfPairs = loadOnehotDict(Overlord.flickr30kResources + "hist_subjOfPair_ordered.csv", 1);
        _objOfPairs = loadOnehotDict(Overlord.flickr30kResources + "hist_objOfPair_ordered.csv", 1);
        _modifierPairs = loadOnehotDict(Overlord.flickr30kResources + "hist_modifierPair.csv", 1);
        _numericPairs = loadOnehotDict(Overlord.flickr30kResources + "hist_numericModifierPair.csv", 1);
        _prepositionPairs = loadOnehotDict(Overlord.flickr30kResources + "hist_prepositionPair.csv", 1);
        _heads = loadOnehotDict(Overlord.flickr30kResources + "hist_head.csv", 1);
        _modifiers = loadOnehotDict(Overlord.flickr30kResources + "hist_modifier.csv", 1);
        _numerics = loadOnehotDict(Overlord.flickr30kResources + "hist_numericModifier.csv", 1);
        _prepositions = loadOnehotDict(Overlord.flickr30kResources + "hist_preposition.csv", 1);
        _types = loadOnehotDict(Overlord.flickr30kResources + "hist_type.csv", 1000);
        _lefts = loadOnehotDict(Overlord.flickr30kResources + "hist_left.csv", 1000);
        _rights = loadOnehotDict(Overlord.flickr30kResources + "hist_right.csv", 1000);
        _subjOfs = loadOnehotDict(Overlord.flickr30kResources + "hist_subjOf.csv", 1);
        _objOfs = loadOnehotDict(Overlord.flickr30kResources + "hist_objOf.csv", 1);
        _pronouns = loadOnehotDict(Overlord.flickr30kResources + "hist_pronoun.csv", 1);
        _pronounTypes = loadOnehotDict(Overlord.flickr30kResources + "hist_pronounType.csv", 0);
        _nonvisuals = loadOnehotDict(Overlord.flickr30kResources + "hist_nonvisual.csv", 1);
        _categories = loadOnehotDict(Overlord.mscocoResources + "hist_cocoCategory.csv", 1000);
        _categoryPairs = loadOnehotDict(Overlord.mscocoResources + "hist_cocoCategoryPair.csv", 1000);

        _hypernyms = new ArrayList<>();
        for(String[] row : FileIO.readFile_table(Overlord.flickr30kResources + "hist_hypernym.csv"))
            _hypernyms.add(row[0]);

        //read other from files
        _colors = new HashSet<>(FileIO.readFile_lineList(Overlord.flickr30kResources + "colors.txt"));
        _stopWords = new HashSet<>(FileIO.readFile_lineList(Overlord.flickr30kResources + "stop_words.txt"));
        List<String> dets = new ArrayList<>();
        _detSet_singular = new HashSet<>();
        _detSet_plural = new HashSet<>();
        String[][] detTable = FileIO.readFile_table(Overlord.flickr30kResources + "dets.csv");
        for (String[] row : detTable) {
            if (row.length > 1) {
                dets.add(row[0]);
                if (row[1].equals("singular")) {
                    _detSet_singular.add(row[0]);
                } else if (row[1].equals("plural")) {
                    _detSet_plural.add(row[0]);
                }
            }
        }
        _dets = new HashMap<>();
        for(int i=0; i<dets.size(); i++)
            _dets.put(dets.get(i), i);

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
                    if (chunkIdx_left > -1 && chunkIdx_left < chunkList.size())
                        chunkNeighbors[0] = chunkList.get(chunkIdx_left);
                    if (chunkIdx_right > -1 && chunkIdx_right < chunkList.size())
                        chunkNeighbors[1] = chunkList.get(chunkIdx_right);

                    _mentionChunkNeighborDict.put(m, chunkNeighbors);
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

        Logger.log("Feature preprocessing (adjacent prepositions)");
        _prepDict_left = new HashMap<>();
        _prepDict_right = new HashMap<>();
        for(Document d : docSet){
            for(Caption c : d.getCaptionList()){
                for(Mention m : c.getMentionList()){
                    List<Chunk> chunkList = m.getChunkList();
                    if(!m.getChunkList().isEmpty()){
                        Chunk left = null;
                        Chunk right = null;
                        if(!chunkList.isEmpty()){
                            left = c.getLeftNeighbor(chunkList.get(0));
                            right = c.getRightNeighbor(chunkList.get(chunkList.size()-1));
                        }
                        if(left != null && left.getChunkType().equals("PP"))
                            _prepDict_left.put(m, left.toString().toLowerCase());
                        if(right != null && right.getChunkType().equals("PP"))
                            _prepDict_right.put(m, right.toString().toLowerCase());
                    }
                }
            }
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

        Logger.log("Feature preprocessing (lemma counts)");
        _imgLemmaCountDict = new HashMap<>();
        for (Document d : docSet) {
            _imgLemmaCountDict.put(d.getID(), new DoubleDict<>());
            for (Mention m : d.getMentionList())
                _imgLemmaCountDict.get(d.getID()).increment(m.getHead().getLemma().toLowerCase().trim());
        }

        Logger.log("Feature preprocessing (hypernyms)");
        _hypDict = new HashMap<>();
        WordnetUtil wnUtil = new WordnetUtil(Overlord.wordnetDir);
        for(Document d : docSet){
            for(Mention m : d.getMentionList()){
                String lemma = m.getHead().getLemma().toLowerCase();
                if(!_hypDict.containsKey(lemma)){
                    Set<String> boh = wnUtil.getBagOfHypernyms(lemma);
                    if(boh.isEmpty())
                        boh.add("");
                    _hypDict.put(lemma, boh);
                }
            }
        }


        _articles = new HashSet<>(Arrays.asList(new String[]{"a", "the", "an"}));
        _prps = new HashSet<>(Arrays.asList(new String[]{"his", "hers", "its", "their"}));
        _masses = new HashSet<>(Arrays.asList(
                new String[]{"sand", "snow", "tea", "water","beer", "coffee",
                        "dirt", "corn", "liquid", "wine"}));
        _collectives = new HashSet<>(
                FileIO.readFile_lineList(
                        Overlord.flickr30kResources + "collectiveNouns.txt", true));
        _portions = new HashSet<>(Arrays.asList(
                new String[]{"pile", "sheet", "puddle", "mound",
                        "spray", "loaf", "cloud", "drink",
                        "sea", "handful", "bale", "line", "row"}));
        _collectives_kv = new HashMap<>();
        _collectives_kv.put("couple", 2);
        _collectives_kv.put("pair", 2);
        _collectives_kv.put("both", 2);
        _collectives_kv.put("either", 2);
        _collectives_kv.put("trio", 3);
        _collectives_kv.put("quartet", 4);
        _collectives_kv.put("dozen", 12);
        _collectives_kv.put("hundred", 100);

        _quantifiers_kv = new HashMap<>();
        _quantifiers_kv.put("several", 3);
        _quantifiers_kv.put("many", 3);
        _quantifiers_kv.put("multiple", 2);
        _quantifiers_kv.put("a few", 2);
        _quantifiers_kv.put("some", 2);

        _distances = new HashMap<>();
        _distances.put("null", 0);
        for(int i=1; i<11; i++)
            _distances.put(String.valueOf(i), i);
        _distances.put(">10", 11);
    }

    /**Exports relation features to outroot.feats, using the given collection of
     * Documents, number of threads, and whether to include the subset and
     * partOf labels (2/3 and 4, respectively)
     *
     * @param docSet        Collection of Documents for which to extract features
     * @param outroot       Root of the file to write features to
     * @param numThreads    Size of the thread pool
     * @param includeSubset Whether to include the subset label (2/3)
     * @param includePartOf Whether to include the partOf label (4)
     */
    public static void exportFeatures_relation(Collection<Document> docSet, String outroot,
                                               int numThreads, boolean forNeural,
                                               boolean includeSubset, boolean includePartOf,
                                               boolean includeCard, String cardFile)
    {
        //One of the problems we've run into is accounting for
        //the exact number of feature vectors we expect, how many we process,
        //and how many we get; we therefore need to keep better track of this
        int numValidMentionPairs = 0;
        for(Document d : docSet){
            List<Mention> mentionList = d.getMentionList();
            for(int i=0; i<mentionList.size(); i++){
                Mention m_i = mentionList.get(i);
                Mention.PRONOUN_TYPE type_i = m_i.getPronounType();
                if(forNeural || type_i == Mention.PRONOUN_TYPE.NONE || type_i == Mention.PRONOUN_TYPE.SEMI){
                    /*
                    if(d.getIsTrain() && m_i.getChainID().equals("0"))
                        continue;*/
                    for(int j=i+1; j<mentionList.size(); j++){
                        Mention m_j = mentionList.get(j);
                        Mention.PRONOUN_TYPE type_j = m_j.getPronounType();
                        if(forNeural || type_j == Mention.PRONOUN_TYPE.NONE || type_j == Mention.PRONOUN_TYPE.SEMI){
                            /*if(d.getIsTrain() && m_j.getChainID().equals("0"))
                                continue;*/
                            numValidMentionPairs += 2;
                        }
                    }
                }
            }
        }
        int numFeatureVectors = 0;


        //Feature preprocessing
        _featurePreprocessing(docSet);

        //Open the feature file for writing
        _outroot = outroot + ".feats";
        Logger.log("Opening [" +_outroot+ "] for writing");
        BufferedWriter bw = null;
        try {
            bw = new BufferedWriter(new FileWriter(_outroot));
        } catch(IOException ioEx) {
            System.err.println("Could not save output file " + _outroot);
            System.exit(0);
        }

        //Put the documents in an ordered list and initialize the thread pool
        List<Document> docList = new ArrayList<>(docSet);
        RelationExtractionThread[] threadPool = new RelationExtractionThread[numThreads];

        //Read cardinality scores, if we have them
        Map<String, Map<String, double[]>> cardinalityScores = new HashMap<>();
        if(includeCard && !docList.get(0).getIsTrain()){
            Logger.log("Reading cardinality scores");
            Map<String, double[]> cardScores = readMccScoresFile(cardFile);

            //As a pre-processing step, split the scores
            //by document so the threads don't need to carry
            //the entire list
            for(String mentionID : cardScores.keySet()){
                String docID = mentionID.split("#")[0];
                if(!cardinalityScores.containsKey(docID))
                    cardinalityScores.put(docID, new HashMap<>());
                cardinalityScores.get(docID).put(mentionID, cardScores.get(mentionID));
            }
        }

        //create and start our first set of threads
        int docIdx = 0;
        for(int i=0; i<numThreads; i++) {
            Document d = docList.get(docIdx);
            threadPool[i] = new RelationExtractionThread(d, forNeural,
                    includeSubset, includePartOf, includeCard,
                    cardinalityScores.get(d.getID()));
            docIdx++;
        }


        for(int i=0; i<threadPool.length; i++)
            threadPool[i].start();
        boolean foundLiveThread = true;
        Set<String> addedFeatureVectors = new HashSet<>();
        DoubleDict<Integer> labelDistro = new DoubleDict<>();
        while(docIdx < docList.size() || foundLiveThread) {
            foundLiveThread = false;
            for(int i=0; i<numThreads; i++) {
                if(threadPool[i].isAlive()) {
                    foundLiveThread = true;
                } else {
                    //if this is a dead thread, store it
                    try{
                        for(FeatureVector fv : threadPool[i].fvSet) {
                            //UPDATE: For reasons I don't quite understand, we can't
                            //simply not add threads for documents we've already added
                            //(there's some duplication happening) so now, at the vector level,
                            //we only add new vectors; this has removed our dup problem

                            if(!addedFeatureVectors.contains(fv.comments)){
                                bw.write(fv.toString() + "\n");
                                labelDistro.increment((int)fv.label);
                                numFeatureVectors++;
                                addedFeatureVectors.add(fv.comments);
                            }
                        }
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
                        threadPool[i] = new RelationExtractionThread(d, forNeural,
                                includeSubset, includePartOf, includeCard,
                                cardinalityScores.get(d.getID()));
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
            try{
                for(FeatureVector fv : threadPool[i].fvSet) {
                    if(!addedFeatureVectors.contains(fv.comments)){
                        bw.write(fv.toString() + "\n");
                        labelDistro.increment((int)fv.label);
                        numFeatureVectors++;
                        addedFeatureVectors.add(fv.comments);
                    }
                }
            } catch(IOException ioEx){
                Logger.log(ioEx);
            }
        }

        Logger.log("Label distro");
        System.out.print(labelDistro.toString());
        Logger.log("Feature vectors");
        System.out.println("Mention Pairs:   " + numValidMentionPairs);
        System.out.println("Feature Vectors: " + numFeatureVectors);
        System.out.println("Labels:          " + labelDistro.getSum());

        Logger.log("Closing [" + _outroot + "]");
        try {
            bw.close();
        } catch(IOException ioEx) {
            System.err.println("Could not save output file " + _outroot);
        }
        JsonIO.writeFile(RelationExtractionThread.getMetaDict(), outroot + "_meta", false);
    }

    /**Exports nonvisual features to [outroot].feats, using the given docSet
     *
     * @param docSet
     * @param outroot
     */
    public static void exportFeatures_nonvis(Collection<Document> docSet, String outroot,
                                             boolean forNeural, boolean includeCard,
                                             String cardFile)
    {
        _exportFeatures_singleMention(docSet, outroot, "nonvis", forNeural,
                                      includeCard, cardFile);
    }

    /**Exports box cardinality features to [outroot].feats, using the given docSet
     *
     * @param docSet
     * @param outroot
     */
    public static void exportFeatures_cardinality(Collection<Document> docSet,
                                                  String outroot, boolean forNeural)
    {
        _exportFeatures_singleMention(docSet, outroot, "cardinality",
                                      forNeural, false, null);
    }

    public static void exportFeatures_combined(Collection<Document> docSet, String outRoot)
    {
        _exportFeatures_singleMention(docSet, outRoot, "combined", false, false, null);
    }

    public static void exportFeatures_phase1(Collection<Document> docSet, String outRoot)
    {
        _exportFeatures_singleMention(docSet, outRoot, "phase_1", false, false, null);
    }


    /**Exports features representing a single mention (since we're double dipping with some of these);
     * labelType \in {'nonvis', 'boxCard'}
     *
     * @param docSet
     * @param outroot
     * @param labelType
     */
    private static void _exportFeatures_singleMention(Collection<Document> docSet,
                                                      String outroot, String labelType,
                                                      boolean forNeural, boolean includeCard,
                                                      String cardFile)
    {
        //Feature preprocessing
        _featurePreprocessing(docSet);

        //Hacky way to grab which document set we're working with
        boolean isTrainSet = false;
        for(Document d : docSet){
            isTrainSet = d.getIsTrain();
            break;
        }

        //Read cardinality scores, if we have them
        Map<String, double[]> cardScores = new HashMap<>();
        if(includeCard && !isTrainSet){
            Logger.log("Reading cardinality scores");
            cardScores = readMccScoresFile(cardFile);
        }

        Map<String, Object> metaDict = new HashMap<>();

        Logger.log("Extracting features");
        Set<FeatureVector> fvSet = new HashSet<>();
        for(Document d : docSet) {
            for (Mention m : d.getMentionList()) {
                //if this is an unreviewed pronominal mention during nonvis train,
                //skip it (since these annotations are unreliable)
                if(labelType.equals("nonvis") && !d.reviewed &&
                   m.getPronounType() != Mention.PRONOUN_TYPE.NONE)
                    continue;

                FeatureVector fv = new FeatureVector();
                fv.comments = m.getUniqueID();
                switch(labelType){
                    case "nonvis": fv.label = m.getChainID().equals("0") ? 1.0 : 0.0;
                        break;
                    case "cardinality": fv.label = Math.min(d.getBoxSetForMention(m).size(), 11);
                        break;
                    case "combined":
                        fv.label = m.getChainID().equals("0") ? -1.0 :
                                Math.min(d.getBoxSetForMention(m).size(), 11);
                        break;
                    case "phase_1":
                        if(m.getChainID().equals("0"))
                            fv.label = 0.0;
                        else if(d.getBoxSetForMention(m).isEmpty())
                            fv.label = 1.0;
                        else
                            fv.label = 2.0;
                        break;
                }
                int currentIdx = 1;

                //head word / modifiers / lexical type
                String head = m.getHead().toString().toLowerCase();
                String[] mods = m.getModifiers();
                String lexType = m.getLexicalType().toLowerCase();
                String cocoCat = Mention.getLexicalEntry_cocoCategory(m, true);
                String lemma = m.getHead().getLemma().toLowerCase();
                if(!forNeural){
                    currentIdx = _addOneHotVector(head, fv, currentIdx, _heads, "head_onehot", metaDict);
                    currentIdx = _addOneHotVector(mods[0], fv, currentIdx, _numerics, "numeric_onehot", metaDict);
                    currentIdx = _addOneHotVector(mods[1], fv, currentIdx, _modifiers, "modifier_onehot", metaDict);
                    currentIdx = _addOneHotVector(lemma, fv, currentIdx, _nonvisuals, "nonvisual_lemma_onehot", metaDict);
                }
                currentIdx = _addOneHotVector(lexType, fv, currentIdx, _types, "lexical_type_onehot", metaDict);
                currentIdx = _addOneHotVector(cocoCat, fv, currentIdx, _categories, "coco_category_onehot", metaDict);

                //governing verbs
                Chunk subjOf = _subjOfDict.get(m); String subjOfStr = "";
                if(subjOf != null)
                    subjOfStr = subjOf.getTokenList().get(subjOf.getTokenList().size()-1).toString().toLowerCase();
                Chunk objOf = _objOfDict.get(m); String objOfStr = "";
                if(objOf != null)
                    objOfStr = objOf.getTokenList().get(objOf.getTokenList().size()-1).toString().toLowerCase();
                if(!forNeural){
                    currentIdx = _addOneHotVector(subjOfStr, fv, currentIdx, _subjOfs, "subj_of_onehot", metaDict);
                    currentIdx = _addOneHotVector(objOfStr, fv, currentIdx, _objOfs, "obj_of_onehot", metaDict);
                }

                //right and left chunk types
                Chunk[] chunkNeighbors = _mentionChunkNeighborDict.get(m);
                String leftChunkType = "", rightChunkType = "";
                if(chunkNeighbors != null && chunkNeighbors[0] != null)
                    leftChunkType = chunkNeighbors[0].getChunkType();
                if(chunkNeighbors != null && chunkNeighbors[1] != null)
                    rightChunkType = chunkNeighbors[1].getChunkType();
                currentIdx = _addOneHotVector(leftChunkType, fv,
                        currentIdx, _lefts, "left_chunk_type_onehot", metaDict);
                currentIdx = _addOneHotVector(rightChunkType, fv,
                        currentIdx, _rights, "right_chunk_type_onehot", metaDict);

                //pronouns
                String pronomText = m.toString().toLowerCase();
                currentIdx = _addOneHotVector(pronomText, fv, currentIdx, _pronouns, "pronoun_onehot", metaDict);
                String pronomType = m.getPronounType().toString();
                currentIdx = _addOneHotVector(pronomType, fv, currentIdx, _pronounTypes, "pronoun_type_onehot", metaDict);

                //prepositions
                String leftPrep = _prepDict_left.containsKey(m) ? _prepDict_left.get(m) : "";
                String rightPrep = _prepDict_right.containsKey(m) ? _prepDict_right.get(m) : "";
                currentIdx = _addOneHotVector(leftPrep, fv, currentIdx,
                        _prepositions, "left_preposition_onehot", metaDict);
                currentIdx = _addOneHotVector(rightPrep, fv, currentIdx,
                        _prepositions, "right_preposition_onehot", metaDict);

                //hypernyms
                if(!forNeural){
                    Set<String> hypSet = _hypDict.get(m.getHead().getLemma().toLowerCase());
                    int start = currentIdx;
                    for(String hyp : _hypernyms){
                        if(hypSet != null && hypSet.contains(hyp))
                            fv.addFeature(currentIdx, 1.0);
                        currentIdx++;
                    }
                    _addMetaEntry("hypernym_bow", start, currentIdx, metaDict);
                }

                //new subset features
                int f_hasArticle = _articles.contains(m.getTokenList().get(0).toString().toLowerCase()) ? TRUE : FALSE;
                fv.addFeature(currentIdx, f_hasArticle);
                _addMetaEntry("hasArticle", currentIdx, metaDict);
                currentIdx++;
                int f_hasMass = _masses.contains(m.getTokenList().get(0).toString().toLowerCase()) ? TRUE : FALSE;
                fv.addFeature(currentIdx, f_hasMass);
                _addMetaEntry("hasMass", currentIdx, metaDict);
                currentIdx++;
                int f_hasCollective = FALSE;
                for(Token t : m.getTokenList())
                    if(_collectives.contains(t.toString().toLowerCase()) || _collectives.contains(t.getLemma()))
                        f_hasCollective = TRUE;
                fv.addFeature(currentIdx, f_hasCollective);
                _addMetaEntry("hasCollective", currentIdx, metaDict);
                currentIdx++;
                int f_hasPortion = FALSE;
                for(Token t : m.getTokenList())
                    if(_portions.contains(t.toString().toLowerCase()) || _portions.contains(t.getLemma()))
                        f_hasPortion = TRUE;
                fv.addFeature(currentIdx, f_hasPortion);
                _addMetaEntry("hasPortion", currentIdx, metaDict);
                currentIdx++;
                int f_isSingular = m.getHead().getPosTag().equals("NN") ||
                        m.getHead().getPosTag().equals("NNP") ? TRUE : FALSE;
                int f_isPlural = m.getHead().getPosTag().equals("NNS") ||
                        m.getHead().getPosTag().equals("NNPS") ? TRUE : FALSE;
                fv.addFeature(currentIdx, f_isSingular);
                _addMetaEntry("isSingular", currentIdx, metaDict);
                currentIdx++;
                fv.addFeature(currentIdx, f_isPlural);
                _addMetaEntry("isPlural", currentIdx, metaDict);
                currentIdx++;
                int f_isSemi = m.getPronounType() ==
                        Mention.PRONOUN_TYPE.SEMI ? TRUE : FALSE;
                fv.addFeature(currentIdx, f_isSemi);
                _addMetaEntry("isSemi", currentIdx, metaDict);
                currentIdx++;
                int[] f_knownQuantity = new int[6];
                Arrays.fill(f_knownQuantity, FALSE);
                int knownQuantity = _getKnownQuantity(m);
                if(knownQuantity > 0 && knownQuantity <= 6)
                    f_knownQuantity[knownQuantity-1] = TRUE;
                for(int i=1; i<=6; i++){
                    fv.addFeature(currentIdx, f_knownQuantity[i-1]);
                    _addMetaEntry("knownQuantity_" + i, currentIdx, metaDict);
                    currentIdx++;
                }

                //Add the cardinality scores _if_ specified
                if(includeCard){
                    double[] cardArr = new double[12];
                    if(d.getIsTrain()){
                        //For training documents, we use the
                        //actual cardinality
                        int goldCard = Math.min(d.getBoxSetForMention(m).size(), 11);
                        Arrays.fill(cardArr, 0.0);
                        cardArr[goldCard] = 1.0;
                    } else {
                        //Get the predicted scores for this mention
                        cardArr = cardScores.get(m.getUniqueID());
                    }

                    for(int i=0; i<12; i++){
                        fv.addFeature(currentIdx, cardArr[i]);
                        _addMetaEntry("cardinality_" + i, currentIdx, metaDict);
                        currentIdx++;
                    }
                }

                //Add the max and the feature fector to the set
                _addMetaEntry("max_idx", currentIdx+1, metaDict);
                fvSet.add(fv);
            }
        }

        Logger.log("Writing to " + outroot + ".feats");
        FileIO.writeFile(fvSet, outroot, "feats", false);
        JsonIO.writeFile(metaDict, outroot + "_meta", false);
    }

    /**Reads the given multiclass scores file into a mappng of IDs->scoreDistro;
     * intended for use with pairwise and cardinality scores.
     *
     * @param filename
     * @return
     */
    public static Map<String, double[]> readMccScoresFile(String filename)
    {
        List<String> ll_scores = FileIO.readFile_lineList(filename);
        Map<String, double[]> mccScores = new HashMap<>();
        for(String line : ll_scores) {
            String[] linePars = line.split(",");
            double[] scores = new double[linePars.length-1];
            for(int i=1; i<linePars.length; i++)
                scores[i-1] = Math.exp(Double.parseDouble(linePars[i]));
            mccScores.put(linePars[0], scores);
        }
        return mccScores;
    }

    /**Evalutes the given nonvis scores (produced by a model) against
     * a simple heuristic baseline
     *
     * @param docSet
     * @param nonvisScoresFile
     */
    public static void evaluateNonvis(Collection<Document> docSet, String nonvisScoresFile)
    {
        String nonvisHistFile = Overlord.flickr30kResources + "hist_nonvisual.csv";
        Logger.log("Loading frequent nonvisual head words from " + nonvisHistFile);
        Set<String> freqNonvisHeads = loadOnehotDict(nonvisHistFile, 10).keySet();

        Logger.log("Loading predicted nonvisual score dict from " + nonvisScoresFile);
        Logger.log("WARNING: using new nonvis score file formatting (mcc score format)");
        Map<String, double[]> nonvisScoreDict = ClassifyUtil.readMccScoresFile(nonvisScoresFile);

        /*
        BinaryClassifierScoreDict nonvis_scoreDict =
                new BinaryClassifierScoreDict(nonvisScoresFile);*/

        Logger.log("Evaluating visual mention detection");
        ScoreDict<Integer> scoreDict_heur = new ScoreDict<>();
        ScoreDict<Integer> scoreDict_model = new ScoreDict<>();
        for(Document d : docSet){
            for(Mention m : d.getMentionList()){
                int gold = m.getChainID().equals("0") ? 0 : 1;
                int pred_model = 0;
                if(nonvisScoreDict.containsKey(m.getUniqueID())){
                    double[] scores = nonvisScoreDict.get(m.getUniqueID());
                    pred_model = scores[0] > scores[1] ? 1 : 0;
                }
                /*
                int pred_model = nonvis_scoreDict.get(m) != null &&
                        nonvis_scoreDict.get(m) < 0 ? 1 : 0;*/
                int pred_heur = 1;
                if(freqNonvisHeads.contains(m.getHead().toString().toLowerCase()))
                    pred_heur = 0;
                scoreDict_heur.increment(gold, pred_heur);
                scoreDict_model.increment(gold, pred_model);
            }
        }
        System.out.printf("%12s: %s (Acc: %.2f%%)\n", "Heuristic",
                scoreDict_heur.getScore(1).toScoreString(), scoreDict_heur.getAccuracy());
        System.out.printf("%12s: %s (Acc: %.2f%%)\n", "Model",
                scoreDict_model.getScore(1).toScoreString(), scoreDict_model.getAccuracy());
    }

    /**Evalutes the given affinity scores (produced by a model) against
     * a simple category-matching baseline; strictHeuristic specifies whether
     * to use strict matches for the coco category baseline
     *
     * @param docSet
     * @param affinityScoresFile
     */
    public static void evaluateAffinity_coco(Collection<Document> docSet,
                                             String affinityScoresFile,
                                             String nonvisScoresFile,
                                             boolean strictHeuristic)
    {
        Logger.log("Initializing lexicons");
        Mention.initializeLexicons(Overlord.flickr30k_lexicon,
                Overlord.mscoco_lexicon);

        Logger.log("Loading predicted affinity score dict from " + affinityScoresFile);
        Map<String, double[]> affinityScores =
                ClassifyUtil.readMccScoresFile(affinityScoresFile);
        Map<String, Integer> affinityScoreDict = new HashMap<>();
        affinityScores.forEach((k,v) ->
            affinityScoreDict.put(k, Util.getMaxIdx(ArrayUtils.toObject(v))));

        Logger.log("Loading predicted nonvis scores from " + nonvisScoresFile);
        BinaryClassifierScoreDict nonvis_scoreDict =
                new BinaryClassifierScoreDict(nonvisScoresFile);
        Set<String> nonvisMentions = new HashSet<>();
        for(Document d : docSet) {
            for (Mention m : d.getMentionList()) {
                if (nonvis_scoreDict.get(m) != null &&
                        nonvis_scoreDict.get(m) >= 0) {
                    nonvisMentions.add(m.getUniqueID());
                }
            }
        }


        Logger.log("Evaluating affinity prediction");
        ScoreDict<Integer> scoreDict_heur = new ScoreDict<>();
        ScoreDict<Integer> scoreDict_model = new ScoreDict<>();
        DoubleDict<String> perfectDict = new DoubleDict<>();
        for(Document d : docSet){
            boolean incorrectMention_heur = false;
            boolean incorrectMention_model = false;
            List<Mention> mentions = d.getMentionList();
            Set<BoundingBox> boxes = d.getBoundingBoxSet();
            for(Mention m : mentions){
                String mentionCats = Mention.getLexicalEntry_cocoCategory(m, !strictHeuristic);
                Set<BoundingBox> assocBoxes = d.getBoxSetForMention(m);

                boolean incorrectBox_heur = false;
                boolean incorrectBox_model = false;
                for(BoundingBox b : boxes){
                    String id = m.getUniqueID() + "|" + b.getUniqueID();
                    int gold = assocBoxes.contains(b) ? 1 : 0;
                    int pred_model = 0, pred_heur = 0;
                    if(!nonvisMentions.contains(m.getUniqueID())){
                        if(affinityScoreDict.containsKey(id))
                            pred_model = affinityScoreDict.get(id);
                        if(mentionCats != null && mentionCats.contains(b.getCategory()))
                            pred_heur = 1;
                    }

                    incorrectBox_heur |= gold != pred_heur;
                    incorrectBox_model |= gold != pred_model;

                    scoreDict_heur.increment(gold, pred_heur);
                    scoreDict_model.increment(gold, pred_model);
                }
                if(!incorrectBox_heur)
                    perfectDict.increment("mentions_heur");
                if(!incorrectBox_model)
                    perfectDict.increment("mentions_model");
                incorrectMention_heur |= incorrectBox_heur;
                incorrectMention_model |= incorrectBox_model;
                perfectDict.increment("mentions");
            }
            if(!incorrectMention_heur)
                perfectDict.increment("imgs_heur");
            if(!incorrectMention_model)
                perfectDict.increment("imgs_model");
            perfectDict.increment("imgs");
        }
        System.out.printf("%12s: %s (Acc: %.2f%%)\n", "Heuristic",
                scoreDict_heur.getScore(1).toScoreString(), scoreDict_heur.getAccuracy());
        System.out.printf("Perfect imgs: %d (%.2f%%); mentions: %d (%.2f%%)\n",
                (int)perfectDict.get("imgs_heur"), 100.0 * perfectDict.get("imgs_heur") /
                perfectDict.get("imgs"), (int)perfectDict.get("mentions_heur"),
                100.0 * perfectDict.get("mentions_heur") / perfectDict.get("mentions"));
        System.out.printf("%12s: %s (Acc: %.2f%%)\n", "Model",
                scoreDict_model.getScore(1).toScoreString(), scoreDict_model.getAccuracy());
        System.out.printf("Perfect imgs: %d (%.2f%%); mentions: %d (%.2f%%)\n",
                (int)perfectDict.get("imgs_model"), 100.0 * perfectDict.get("imgs_model") /
                        perfectDict.get("imgs"), (int)perfectDict.get("mentions_model"),
                100.0 * perfectDict.get("mentions_model") / perfectDict.get("mentions"));
    }

    public static void exportStanfordCorefConll(Collection<Document> docSet)
    {
        Logger.log("Initializing Stanford Annotation object");
        StanfordAnnotator stanfordAnno = StanfordAnnotator.createCoreference(false);

        Logger.log("Processing documents; writing .conll files to out/stanford/");
        int docIdx = 0;
        for(Document d : docSet){
            docIdx++;
            Logger.logStatus("Processed %d (%.2f%% docs)", docIdx,
                    100.0 * docIdx / docSet.size());

            //predict the document as a stanford-parsed object
            String text = "";
            for(Caption c : d.getCaptionList())
                text += c.toString() + " ";
            text = text.trim();
            Document d_stanford = stanfordAnno.annotate(d.getID(), text);

            String outDir = "out/stanford/";

            //Write the key file
            List<String> lineList_key = d.toConll2012();
            lineList_key.add(0, "#begin document (" + d.getID() + "); part 000");
            lineList_key.add("#end document");
            FileIO.writeFile(lineList_key, outDir +
                    d.getID().replace(".jpg", "") + "_key", "conll", false);

            //Write the response file
            List<String> lineList_resp = d_stanford.toConll2012();
            lineList_resp.add(0, "#begin document (" + d.getID() + "); part 000");
            lineList_resp.add("#end document");
            FileIO.writeFile(lineList_resp, outDir +
                    d.getID().replace(".jpg", "") + "_response", "conll", false);
        }
    }


    public static void exportPronomConll(Collection<Document> docSet, boolean predictPronom)
    {
        String outDir = "out/pronom/";
        for(Document d : docSet) {
            Set<Chain> predChainSet = pronominalCorefChains(d, predictPronom);

            //Write the key file
            List<String> lineList_key = d.toConll2012();
            lineList_key.add(0, "#begin document (" + d.getID() + "); part 000");
            lineList_key.add("#end document");
            FileIO.writeFile(lineList_key, outDir +
                    d.getID().replace(".jpg", "") + "_key", "conll", false);

            //Write the response file
            List<String> lineList_resp = Document.toConll2012(d, predChainSet);
            lineList_resp.add(0, "#begin document (" + d.getID() + "); part 000");
            lineList_resp.add("#end document");
            FileIO.writeFile(lineList_resp, outDir +
                    d.getID().replace(".jpg", "") + "_response", "conll", false);
        }
    }

    /**Exports the box-mention affinity features into a single file;
     * all vectors are returned for dev/text, but -- for train --
     * a random sampling of 10 mention-box pairs are given, per mention text
     *
     * @param docSet
     * @param dataSplit
     */
    public static void exportFeatures_affinity(Collection<Document> docSet, String dataSplit)
    {
        List<Document> docList = new ArrayList<>(docSet);

        Logger.log("Initializing stop words"); //stop words courtesy of http://www.ranks.nl/
        _stopWords = new HashSet<>(FileIO.readFile_lineList(Overlord.flickr30kResources + "stop_words.txt"));

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
            ll_affinity.addAll(_exportFeatures_affinity_train(docSet, boxFiles));
        } else {
            List<String> ll_types = new ArrayList<>();
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
                        ll_types.add(ID + "," + m.getLexicalType());
                    }
                }
                docIdx++;
                Logger.logStatus("complete (%.2f%%)", 100.0 * (double)docIdx /
                        docSet.size());
            }
            FileIO.writeFile(ll_types, Overlord.dataPath +
                    "feats/affinity_feats_" + dataSplit + "_type", "csv", false);
        }

        FileIO.writeFile(ll_affinity, Overlord.dataPath +
                "feats/affinity_feats_" + dataSplit, "feats", false);
    }

    /**Exports affiity features for train, where 10 random boxes are sampled per
     * unique mention string
     *
     * @param docSet
     * @param boxFiles
     * @return
     */
    private static List<String> _exportFeatures_affinity_train(Collection<Document> docSet, Set<String> boxFiles)
    {
        List<String> ll_affinity = new ArrayList<>();

        //Store a mapping of [docID -> [mention -> [bounding boxes] ] ]
        Map<String, Map<Mention, Set<BoundingBox>>>
                mentionBoxesDict = new HashMap<>();
        Map<String, String> typeDict = new HashMap<>();
        Map<Mention, Document> mentionDocDict = new HashMap<>();
        for(Document d : docSet){
            if(boxFiles.contains(d.getID().replace(".jpg", ""))){
                for(Mention m : d.getMentionList()){
                    mentionDocDict.put(m, d);
                    String normText = m.toString().toLowerCase().trim();
                    if(!mentionBoxesDict.containsKey(normText)) {
                        mentionBoxesDict.put(normText, new HashMap<>());
                        typeDict.put(normText, m.getLexicalType());
                    }
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
                String ID = typeDict.get(normText); //add the type instead of the ID, because train has no ID
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
                String ID = typeDict.get(normText); //add the type instead of the ID, because train has no ID
                ll_affinity.add(ID + ",0," + StringUtil.listToString(feats_affinity, ","));
            }
        }
        return ll_affinity;
    }

    /**Evaluates the pronominal coreference system, where the score dict
     * is computed over all the links where at least one mention is
     * fully pronominal
     *
     * @param docSet
     * @return
     */
    public static ScoreDict<Integer> evaluatePronomCoref(Collection<Document> docSet,
         BinaryClassifierScoreDict nonvisualScoreDict)
    {
        ScoreDict<Integer> scores = new ScoreDict<>();
        for(Document d : docSet){
            List<Mention> mentions = d.getMentionList();

            //get the set of visual mentions (either predicted or gold)
            Set<Mention> visualMentions = new HashSet<>();
            for(Mention m : mentions){
                boolean isNonvis = m.getChainID().equals("0");
                if(nonvisualScoreDict != null){
                    Double nonvisScore = nonvisualScoreDict.get(m);
                    isNonvis = nonvisScore != null && nonvisScore > 0;
                }
                if(!isNonvis)
                    visualMentions.add(m);
            }

            Set<String> pronomCoref = pronominalCoref(d, visualMentions);
            for(int i=0; i<mentions.size(); i++) {
                Mention m_i = mentions.get(i);
                Mention.PRONOUN_TYPE pType_i = m_i.getPronounType();
                for (int j = i + 1; j < mentions.size(); j++) {
                    Mention m_j = mentions.get(j);
                    Mention.PRONOUN_TYPE pType_j = m_j.getPronounType();

                    //We only care about same-caption links where one end is a pronoun
                    if(m_i.getCaptionIdx() != m_j.getCaptionIdx())
                        continue;
                    if(pType_i != Mention.PRONOUN_TYPE.NONE && pType_i != Mention.PRONOUN_TYPE.SEMI ||
                       pType_j != Mention.PRONOUN_TYPE.NONE && pType_j != Mention.PRONOUN_TYPE.SEMI){
                        int gold = !m_i.getChainID().equals("0") &&
                                   m_i.getChainID().equals(m_j.getChainID()) ? 1 : 0;

                        String id_ij = Document.getMentionPairStr(m_i, m_j);
                        String id_ji = Document.getMentionPairStr(m_j, m_i);
                        int pred = pronomCoref.contains(id_ij) ||
                                   pronomCoref.contains(id_ji) ? 1 : 0;

                        scores.increment(gold, pred);
                    }
                }
            }
        }
        return scores;
    }

    /**Performs heuristic pronominal coreference resolution, according
     * to the following:
     *
     * 1) Subject / Object pronouns
     *      - Attach to the furthest antecedent; preference to animate
     * 2) Reflexive / Reciprocal
     *      - Attach to the nearest antecedent; preference to animate
     * 3) Relative
     *      - Exclude X if Y in XofY; attach to nearest
     *      - Attach to S if in [R] [to be / like] [S] construction
     * 4) Indefinite
     *      - No attachment
     * 5) "both", "all", and "it" refer to the nearest antecedent
     *
     * @param d
     * @return
     */
    public static Set<String> pronominalCoref(Document d, Collection<Mention> visualMentions)
    {
        String[] singMods = {"one", "1", "first", "second", "third"};
        String[] undefPronom = {"that", "which", "who", "whom", "what"};
        String[] identityTerms = {"to be", "like"};

        Set<String> corefPairs = new HashSet<>();
        for(Caption c : d.getCaptionList()) {

            List<Mention> mentions = c.getMentionList();
            for(int i=0; i<mentions.size(); i++){
                Mention m_pronom = mentions.get(i);
                Mention.PRONOUN_TYPE pronounType = m_pronom.getPronounType();
                Boolean pluralPronom =
                        Mention.PRONOUN_TYPE.getIsPlural(m_pronom.toString().toLowerCase());

                //if this isn't a full pronoun, we need not concern ourselves
                if(pronounType == Mention.PRONOUN_TYPE.NONE || pronounType == Mention.PRONOUN_TYPE.SEMI)
                    continue;

                //Now that we know m_i is a pronoun, we're going to collect
                //all mentions (left and right) that could be candidates
                List<Mention> leftCandidates = new ArrayList<>(), rightCandidates = new ArrayList<>();
                for(int j=0; j<mentions.size(); j++){
                    if(j == i)
                        continue;

                    Mention m_j = mentions.get(j);

                    //if this isn't one of the visual mentions, bail
                    if(!visualMentions.contains(m_j))
                        continue;

                    //If we've already added this pairing, ignore it now
                    String id_ij = Document.getMentionPairStr(m_pronom, m_j);
                    String id_ji = Document.getMentionPairStr(m_j, m_pronom);
                    if(corefPairs.contains(id_ij) || corefPairs.contains(id_ji))
                        continue;

                    //Only add non-pronominal mentions
                    if(m_j.getPronounType() != Mention.PRONOUN_TYPE.NONE)
                        continue;

                    //Determine mention j's plurality
                    String[] mods = m_j.getModifiers();
                    boolean plural_j = false;
                    if(m_j.getHead().getPosTag().equals("NNS") ||
                       m_j.getHead().getPosTag().equals("NNPS")){
                        plural_j = true;
                    } else if(!mods[0].isEmpty() &&
                              !Arrays.asList(singMods).contains(mods[0])) {
                        plural_j = true;
                    }
                    if(getGenderMatch(m_pronom, m_j)){
                        //The underdefined pronouns match everything; other candidates
                        //must have matching plurality
                        if(Arrays.asList(undefPronom).contains(m_pronom.toString().toLowerCase()) ||
                           pluralPronom != null && (pluralPronom && plural_j || !pluralPronom && !plural_j)){
                            if(j < i)
                                leftCandidates.add(m_j);
                            else if(j > i)
                                rightCandidates.add(m_j);
                        }
                    }
                }

                Mention[] pair = null;

                /* 1) Subjective / Objective */
                if ((pronounType == Mention.PRONOUN_TYPE.SUBJECTIVE_SINGULAR ||
                    pronounType == Mention.PRONOUN_TYPE.SUBJECTIVE_PLURAL ||
                    pronounType == Mention.PRONOUN_TYPE.OBJECTIVE_SINGULAR ||
                    pronounType == Mention.PRONOUN_TYPE.OBJECTIVE_PLURAL) &&
                    !leftCandidates.isEmpty()) {

                    //Furthest antecedent is left, index 0
                    Mention ante = leftCandidates.get(0);
                    //Prefer people/animal antecedents, where available
                    for(int j=0; j<leftCandidates.size(); j++){
                        Mention m_j = leftCandidates.get(j);
                        if(m_j.getLexicalType().contains("people") || m_j.getLexicalType().contains("animals")){
                            ante = m_j; break;
                        }
                    }
                    pair = new Mention[]{ante, m_pronom};
                } /* 2) Reflexive */
                else if ((pronounType == Mention.PRONOUN_TYPE.REFLEXIVE_SINGULAR ||
                        pronounType == Mention.PRONOUN_TYPE.REFLEXIVE_PLURAL ||
                        pronounType == Mention.PRONOUN_TYPE.RECIPROCAL) &&
                        !leftCandidates.isEmpty()) {
                    //Nearest antecedent is left, max idx
                    Mention ante = leftCandidates.get(leftCandidates.size()-1);
                    //Prefer people/animal antecedents, where available
                    for(int j=leftCandidates.size()-1; j>= 0; j--){
                        Mention m_j = leftCandidates.get(j);
                        if(m_j.getLexicalType().contains("people") || m_j.getLexicalType().contains("animals")){
                            ante = m_j; break;
                        }
                    }
                    pair = new Mention[]{ante, m_pronom};
                } /* 3) Relative */
                else if (pronounType == Mention.PRONOUN_TYPE.RELATIVE || m_pronom.toString().equals("that")) {
                    Mention m_left = null;
                    List<Token> interstl_left = new ArrayList<>();
                    if(!leftCandidates.isEmpty()){
                        m_left = leftCandidates.get(leftCandidates.size()-1);
                        interstl_left = c.getInterstitialTokens(m_left, m_pronom);
                    }
                    Mention m_right = null;
                    List<Chunk> interstl_right = new ArrayList<>();
                    if(!rightCandidates.isEmpty()){
                        m_right = rightCandidates.get(0);
                        interstl_right = c.getInterstitialChunks(m_pronom, m_right);
                    }

                    //If this is Y in XofY, drop X
                    if(interstl_left.size() == 1 && interstl_left.get(0).toString().equals("of") &&
                       leftCandidates.size() > 1){
                        m_left = leftCandidates.get(leftCandidates.size()-2);
                    }

                    //Determine if this is one of our identity cases
                    boolean identityCase = false;
                    if(interstl_right.size() == 1){
                        String chunkType = interstl_right.get(0).getChunkType();
                        String chunkStr = interstl_right.get(0).toString().toLowerCase();
                        identityCase = chunkType.equals("VP") &&
                                (StringUtil.containsElement(Arrays.asList(identityTerms), chunkStr) ||
                                chunkStr.equals("is") || chunkStr.equals("are"));
                    }

                    //Either attach on the right because we're an identity case,
                    //or attach left if we have a valid antecedent
                    if(identityCase){
                        pair = new Mention[]{m_pronom, m_right};
                    } else if(m_left != null) {
                        pair = new Mention[]{m_left, m_pronom};
                    }
                } /* 4) Other */
                else if (pronounType == Mention.PRONOUN_TYPE.OTHER && !leftCandidates.isEmpty()) {
                    pair = new Mention[]{leftCandidates.get(leftCandidates.size()-1), m_pronom};
                }

                if(pair != null){
                    corefPairs.add(Document.getMentionPairStr(pair[0], pair[1]));
                    corefPairs.add(Document.getMentionPairStr(pair[1], pair[0]));
                }
            }
        }
        return corefPairs;
    }

    public static Set<Chain> pronominalCorefChains(Document d, boolean predictPronom)
    {
        Map<Mention, String> mentionChainDict = new HashMap<>();

        //Get the visual, nonpronominal mentions for this document
        List<Mention> mentions = d.getMentionList();
        Set<Mention> visualMentions = new HashSet<>();
        for(Mention m : mentions)
            if(m.getPronounType() == Mention.PRONOUN_TYPE.NONE && !m.getChainID().equals("0"))
                visualMentions.add(m);

        //Perform coref
        if(predictPronom){
            Set<String> pronomCorefPairs = ClassifyUtil.pronominalCoref(d, visualMentions);

            //Get the actual chain associations, given those pairs
            for(int i=0; i<mentions.size(); i++){
                Mention m_i = mentions.get(i);
                for(int j=i+1; j<mentions.size(); j++){
                    Mention m_j = mentions.get(j);
                    String id_ij = Document.getMentionPairStr(m_i, m_j);
                    String id_ji = Document.getMentionPairStr(m_j, m_i);
                    if(pronomCorefPairs.contains(id_ij) || pronomCorefPairs.contains(id_ji)) {
                        if(m_i.getPronounType() != Mention.PRONOUN_TYPE.NONE &&
                                m_i.getPronounType() != Mention.PRONOUN_TYPE.SEMI)
                            mentionChainDict.put(m_i, m_j.getChainID());
                        else
                            mentionChainDict.put(m_j, m_i.getChainID());
                    }
                }
            }
        }

        //For every nonpronominal mention, add its chain; unassociated
        //pronouns should have no chain (for this comparison)
        for(int i=0; i<mentions.size(); i++){
            Mention m = mentions.get(i);
            if(!mentionChainDict.containsKey(m) && !m.getChainID().equals("0") &&
                    (m.getPronounType() == Mention.PRONOUN_TYPE.SEMI ||
                            m.getPronounType() == Mention.PRONOUN_TYPE.NONE)){
                mentionChainDict.put(m, m.getChainID());
            }
        }

        //Construct predicted chains
        Map<String, Set<Mention>> predChainDict = Util.invertMap(mentionChainDict);
        Set<Chain> predChainSet = new HashSet<>();
        for(String chainID : predChainDict.keySet()){
            Chain c = new Chain(d.getID(), chainID);
            for(Mention m : predChainDict.get(chainID))
                c.addMention(m);
            predChainSet.add(c);
        }
        return predChainSet;
    }

    /**Performs attribute attachment, associating animate mentions (as keys) with
     * attributes in the form of Annotation objects (Mentions, Chunks, or Tokens)
     *
     * @param docSet
     * @return
     */
    public static Map<Mention, AttrStruct> attributeAttachment_agent(Collection<Document> docSet)
    {
        _featurePreprocessing(docSet);
        Map<Mention, AttrStruct> attributeDict = new HashMap<>();

        Logger.log("Loading frequent heads / attribute locations from files");
        String[][] clothAttrTable = FileIO.readFile_table(Overlord.flickr30kResources + "hist_clothHead.csv");
        Map<String, String> clothAttrs = new HashMap<>();
        for (String[] row : clothAttrTable)
            if(row.length > 2)
                clothAttrs.put(row[0], row[1]);
        String[][] bodypartAttrTable = FileIO.readFile_table(Overlord.flickr30kResources + "hist_bodypartHead.csv");
        Map<String, String> bodypartAttrs = new HashMap<>();
        for(String[] row : bodypartAttrTable)
            if(row.length > 2)
                bodypartAttrs.put(row[0], row[1]);

        Logger.log("Storing mention genders");
        Map<Mention, String> genderDict = new HashMap<>();
        for (Document d : docSet) {
            for (Mention m : d.getMentionList()) {
                Set<String> hyps = _hypDict.get(m.getHead().getLemma());
                String gender = m.getGender(hyps);
                //String gender = m.getGender();
                if (!gender.equals("neuter"))
                    genderDict.put(m, gender);
            }
        }

        Logger.log("Associating bodyparts with agent mentions");
        for (Document d : docSet) {
            for (Caption c : d.getCaptionList()) {
                List<Mention> bodypartList = new ArrayList<>();
                List<Mention> agentList = new ArrayList<>();
                for (Mention m : c.getMentionList()) {
                    if (m.getLexicalType().equals("people") ||
                        m.getLexicalType().equals("animals")) {
                        agentList.add(m);
                    } else if (m.getPronounType().isAnimate()) {
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
                        int maxIdx = Integer.MIN_VALUE;
                        int minIdx = Integer.MAX_VALUE;
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
                        //   X in an XofY construction (ie "the arm of a man")
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
                                    attributeDict.put(agent, _toAttrStruct(agent));

                                String partsHead = m_parts.getHead().getLemma().toLowerCase();
                                String partsNormText = m_parts.toString().toLowerCase();

                                String attrClass = "bodypart";
                                if(bodypartAttrs.containsKey(partsHead))
                                    if(!bodypartAttrs.get(partsHead).equals("none"))
                                        attrClass = bodypartAttrs.get(partsHead);

                                if (partsNormText.contains("her ")) {
                                    attributeDict.get(agent).clearAttribute("gender");
                                    attributeDict.get(agent).addAttribute("gender", "female");
                                } else if (partsNormText.contains("his ")) {
                                    attributeDict.get(agent).clearAttribute("gender");
                                    attributeDict.get(agent).addAttribute("gender", "male");
                                }
                                attributeDict.get(agent).addAttribute(attrClass, _toAttrStruct(m_parts));
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
                    if (m.getLexicalType().equals("people") ||
                        m.getLexicalType().equals("animals")) {
                        agentList.add(m);
                    } else if (m.getPronounType().isAnimate()) {
                        agentList.add(m);
                    } else if (m.getLexicalType().equals("clothing")) {
                        clothingList.add(m);
                    } else if (m.getLexicalType().equals("colors")) {
                        clothingList.add(m);
                    } else if(m.getLexicalType().equals("clothing/colors")){
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
                                attributeDict.put(agent, _toAttrStruct(agent));
                            for (Mention cm : clothCluster) {
                                String clothNormText = cm.toString().toLowerCase();
                                String clothHead = cm.getHead().getLemma().toLowerCase();

                                String attrClass = "clothing";
                                if(clothAttrs.containsKey(clothHead))
                                    if(!clothAttrs.get(clothHead).equals("none"))
                                        attrClass = clothAttrs.get(clothHead);

                                if (clothNormText.contains("her ")) {
                                    attributeDict.get(agent).clearAttribute("gender");
                                    attributeDict.get(agent).addAttribute("gender", "female");
                                } else if (clothNormText.contains("his ")) {
                                    attributeDict.get(agent).clearAttribute("gender");
                                    attributeDict.get(agent).addAttribute("gender", "male");
                                }
                                attributeDict.get(agent).addAttribute(attrClass, _toAttrStruct(cm));
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
                                    attributeDict.put(m, _toAttrStruct(m));
                                String attrName = "modifier";
                                if (_colors.contains(nextChunk.toString().toLowerCase()))
                                    attrName = "color";
                                attributeDict.get(m).addAttribute(attrName, _toAttrStruct(nextChunk));
                            }
                        }
                    }
                }
            }
        }

        return attributeDict;
    }

    private static AttrStruct _toAttrStruct(Annotation core) {
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
            //attr.addAttribute("cardinality", m.getCardinality().toString().replace("|", ","));
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

    public static List<List<Mention>>
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

    /**Returns an alphabetized list of strings in the given filename
     * that occur more than thresh times; originally implemented to
     * help produce one-hot vectors
     *
     * @param filename
     * @param thresh
     * @return
     */
    private static List<String> _getOneHotList(String filename, int thresh)
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

    /**Adds the given item to the given feature vector as
     * a one-hot vector representation, such that the index
     * idxOffset + idxDict[item] = 1; Also adds the range of
     * indices represented by the one-hot vector to the metaDict
     * with the given feature name
     *
     * @param item
     * @param fv
     * @param idxOffset
     * @param idxDict
     * @param featName
     * @param metaDict
     * @return The new idxOffset (idxOffset + idxDict.size())
     */
    private static int _addOneHotVector(String item, FeatureVector fv,
                                        int idxOffset, Map<String, Integer> idxDict,
                                        String featName, Map<String, Object> metaDict)
    {
        //Add this item to the vector (if it's present)
        int end = idxOffset + idxDict.size() + 1;
        if(idxDict.containsKey(item))
            fv.addFeature(idxOffset + idxDict.get(item) + 1, 1.0);

        //Add this onehot to the meta dict
        _addMetaEntry(featName, idxOffset, end, metaDict);

        return end+1;
    }

    /**Adds a (featName,idx) pair to the given metaDict
     *
     * @param featName
     * @param idx
     * @param metaDict
     */
    private static void _addMetaEntry(String featName, int idx,
                                      Map<String, Object> metaDict)
    {
        if(!metaDict.containsKey(featName))
            metaDict.put(featName, idx);
    }

    /**Adds a (featName,[start,end]) pair to the given metaDict
     *
     * @param featName
     * @param startIdx
     * @param endIdx
     * @param metaDict
     */
    private static void _addMetaEntry(String featName, int startIdx, int endIdx,
                                      Map<String, Object> metaDict)
    {
        if(!metaDict.containsKey(featName))
            metaDict.put(featName, new Integer[]{startIdx, endIdx});
    }

    /**Returns the sum of all mentioned numbers (as collectives,
     * numerals, or quantifiers) in the mention; NOTE: this is
     * particularly crude, but works in most cases
     *
     * @param m
     * @return
     */
    private static int _getKnownQuantity(Mention m)
    {
        int knownQuantity = 0;
        for(Token t : m.getTokenList()){
            String text = t.toString().toLowerCase();
            String lemma = t.getLemma();

            if(_collectives_kv.containsKey(text))
                knownQuantity += _collectives_kv.get(text);
            else if(_collectives_kv.containsKey(lemma))
                knownQuantity += _collectives_kv.get(lemma);
            else if(_quantifiers_kv.containsKey(text))
                knownQuantity += _quantifiers_kv.get(text);
            else if(_quantifiers_kv.containsKey(lemma))
                knownQuantity += _quantifiers_kv.get(lemma);
        }
        List<Integer> valList = new ArrayList<>();
        String[] words = m.toString().replace("-", " - ").split(" ");
        for(int i=0; i<words.length; i++)
            valList.add(Util.parseInt(words[i]));
        for(int i=0; i<words.length; i++)
            if(valList.get(i) != null)
                knownQuantity += valList.get(i);
        return knownQuantity;
    }

    /**In order to (drastically) improve relation feature extraction
     * speed, we parallelize this operation giving each document
     * their own thread
     *
     */
    private static class RelationExtractionThread extends Thread
    {
        private static Map<String, Object> _metaDict = new HashMap<>();
        private static final String[] _identityTerms = {"to be", "like"};
        private Document _doc;
        private Set<String> _subsetMentions;
        private Set<String> _partOfMentions;
        private Map<String, double[]> _cardScores;
        private boolean _includeCard;
        private boolean _forNeural;
        private Map<String, Set<Integer>> _chainBoxDict;
        private Map<String, Set<String>> _subsetChainDict;
        Collection<FeatureVector> fvSet;


        /**Initializes a RelationExtractionThread, which extracts
         * features for relation prediction for the given
         * Document's mentions
         *
         * @param doc           Document for which mention features
         *                      will be extracted
         * @param includeSubset Whether to include the subset label (2/3)
         * @param includePartOf Whether to include the partOf label (4)
         */
        RelationExtractionThread(Document doc, boolean forNeural, boolean includeSubset,
                                 boolean includePartOf, boolean includeCard,
                                 Map<String, double[]> cardScores)
        {
            init(doc, forNeural, includeSubset, includePartOf,
                 includeCard, cardScores);
        }

        RelationExtractionThread(Document doc, boolean forNeural)
        {
            init(doc, forNeural, true, false, false, null);
        }

        private void init(Document doc, boolean forNeural, boolean includeSubset,
                          boolean includePartOf, boolean includeCard,
                          Map<String, double[]> cardScores)
        {
            _doc = doc;
            fvSet = new HashSet<>();
            _subsetMentions = new HashSet<>();
            _partOfMentions = new HashSet<>();
            _includeCard = includeCard;
            _forNeural = forNeural;

            //Vary what we load into memory based on the labeling scheme
            if(includeSubset)
                _subsetMentions = _doc.getSubsetMentions();
            if(includePartOf)
                _partOfMentions = _doc.getPartOfMentions();

            //incorporate cardinality scores, if applicable
            _cardScores = new HashMap<>();
            if(includeCard){
                if(cardScores == null){
                    for(Mention m : _doc.getMentionList()){
                        double[] cardinalities = new double[12];
                        Arrays.fill(cardinalities, 0.0);
                        cardinalities[Math.min(_doc.getBoxSetForMention(m).size(), 11)] = 1.0;
                        _cardScores.put(m.getUniqueID(), cardinalities);
                    }
                } else {
                    _cardScores = cardScores;
                }
            }

            //Associate all chains with their boxes
            _chainBoxDict = new HashMap<>();
            for(Chain c : _doc.getChainSet()){
                _chainBoxDict.put(c.getID(), new HashSet<>());
                for(BoundingBox b : c.getBoundingBoxSet())
                    _chainBoxDict.get(c.getID()).add(b.getIdx());
            }

            //Associate chains with all of their supersets
            _subsetChainDict = new HashMap<>();
            for(Chain[] subsetPair : _doc.getSubsetChains()){
                String subID = subsetPair[0].getID();
                String supID = subsetPair[1].getID();
                if(!_subsetChainDict.containsKey(subID))
                    _subsetChainDict.put(subID, new HashSet<>());
                _subsetChainDict.get(subID).add(supID);
            }
        }

        /**Returns the meta-dict for these relation features
         *
         * @return  Mapping of feature name to feature indices
         */
        public static Map<String, Object> getMetaDict()
        {
            return _metaDict;
        }

        /**Extracts relation features for each ordered pair
         * of mentions in this thread's document
         */
        public void run()
        {
            List<Mention> mentionList = _doc.getMentionList();
            for(int i=0; i<mentionList.size(); i++){
                Mention m_i = mentionList.get(i);
                Mention.PRONOUN_TYPE type_i = m_i.getPronounType();
                if(_forNeural || type_i == Mention.PRONOUN_TYPE.NONE ||
                   type_i == Mention.PRONOUN_TYPE.SEMI){
                    //if this is a training document, don't bother training with
                    //nonvisual mentions
                    /*
                    if(_doc.getIsTrain() && m_i.getChainID().equals("0"))
                        continue;*/

                    for(int j=i+1; j<mentionList.size(); j++){
                        Mention m_j = mentionList.get(j);
                        Mention.PRONOUN_TYPE type_j = m_j.getPronounType();
                        if(_forNeural || type_j == Mention.PRONOUN_TYPE.NONE ||
                           type_j == Mention.PRONOUN_TYPE.SEMI){
                            /*
                            if(_doc.getIsTrain() && m_j.getChainID().equals("0"))
                                continue;
                            */
                            fvSet.add(_getRelationFeatureVector(m_i, m_j));
                            fvSet.add(_getRelationFeatureVector(m_j, m_i));
                        }
                    }
                }
            }
        }


        private int _getLabel(Mention m_i, Mention m_j)
        {
            String chainID_i = m_i.getChainID();
            String chainID_j = m_j.getChainID();

            //Nonvisual mentions have disjoint links
            if(chainID_i.equals("0") || chainID_j.equals("0"))
                return 0;

            //Coreferent mentions are label 1
            if(chainID_i.equals(chainID_j))
                return 1;

            //Subset mentions are 2 and 3, respectively
            Set<String> supChains_i = _subsetChainDict.get(chainID_i);
            Set<String> supChains_j = _subsetChainDict.get(chainID_j);
            if(supChains_i != null && supChains_i.contains(chainID_j))
                return 2;
            if(supChains_j != null && supChains_j.contains(chainID_i))
                return 3;

            /*
            //Complement is a symmetric link between entities that
            //belong to the same superset but are not otherwise
            //related
            if(supChains_i != null && supChains_j != null){
                Set<String> supIntersect = new HashSet<>(supChains_i);
                supIntersect.retainAll(supChains_j);
                if(!supIntersect.isEmpty())
                    return 4;
            }

            //Intersecting links occur when two entities share some boxes and
            //the same lexical type (non pronominal mentions only) but do not
            //share all their boxes
            Set<Integer> boxes_i = _chainBoxDict.get(chainID_i);
            Set<Integer> boxes_j = _chainBoxDict.get(chainID_j);
            //If both these mentions have boxes...
            if(!boxes_i.isEmpty() && !boxes_j.isEmpty()){
                Set<Integer> boxIntersect = new HashSet<>(boxes_i);
                boxIntersect.retainAll(boxes_j);
                //...and their box sets intersect but do not subsume one another...
                if(!boxIntersect.isEmpty() && boxes_i.size() > boxIntersect.size() &&
                   boxes_j.size() > boxIntersect.size()){
                    if(m_i.getPronounType() != Mention.PRONOUN_TYPE.NONE ||
                       m_j.getPronounType() != Mention.PRONOUN_TYPE.NONE ||
                       Mention.getLexicalTypeMatch(m_i, m_j) > 0){
                        return 5;
                    }
                }
            }*/

            //All other cases are disjoint, or 0
            return 0;
        }


        /**Return a complete pairwise feature vector, given the
         * ordered pair of mentions
         *
         * @param m1    First mention in the pair
         * @param m2    Second mention in the pair
         * @return      FeatureVector for the ordered mention pair
         */
        private FeatureVector _getRelationFeatureVector(Mention m1, Mention m2)
        {
            int currentIdx = 1;
            List<Object> featureList = new ArrayList<>();

            //Caption match
            if(!_forNeural){
                Integer f_capMatch = m1.getCaptionIdx() == m2.getCaptionIdx() ? 1 : 0;
                featureList.add(f_capMatch);
                _addMetaEntry("caption_match", currentIdx++, _metaDict);
            }

            //caption match _and_ m_i < m_j / m_j < m_i
            Integer f_ante_ij = 0;
            if(m1.getCaptionIdx() == m2.getCaptionIdx() && m1.getIdx() < m2.getIdx())
                f_ante_ij = 1;
            featureList.add(f_ante_ij);
            _addMetaEntry("precede_ij", currentIdx++, _metaDict);

            //Head matches
            String head_1 = m1.getHead().toString().toLowerCase();
            String head_2 = m2.getHead().toString().toLowerCase();
            String headPos_1 = m1.getHead().getPosTag();
            String headPos_2 = m2.getHead().getPosTag();
            Integer f_headMatch = head_1.equals(head_2) ? TRUE : FALSE;
            Integer f_headPOSMatch = headPos_1.equals(headPos_2) ? TRUE : FALSE;
            featureList.add(f_headMatch);
            _addMetaEntry("head_match", currentIdx++, _metaDict);
            featureList.add(f_headPOSMatch);
            _addMetaEntry("head_pos_match", currentIdx++, _metaDict);

            //Lemma match / substring feat
            String lemma_1 = m1.getHead().getLemma().toLowerCase();
            String lemma_2 = m2.getHead().getLemma().toLowerCase();
            Integer f_lemmaMatch = lemma_1.equals(lemma_2) ? TRUE : FALSE;
            Integer f_substring = lemma_1.contains(lemma_2) ||
                    lemma_2.contains(lemma_1) ? TRUE : FALSE;
            featureList.add(f_lemmaMatch);
            _addMetaEntry("lemma_match", currentIdx++, _metaDict);
            featureList.add(f_substring);
            _addMetaEntry("substring_match", currentIdx++, _metaDict);

            //Extent match
            String extent_1 = m1.toString().replace(head_1, "");
            String extent_2 = m2.toString().replace(head_2, "");
            Integer f_extentMatch = UNK;
            if(!extent_1.isEmpty() || !extent_2.isEmpty())
                f_extentMatch = extent_1.equalsIgnoreCase(extent_2) ? TRUE : FALSE;
            featureList.add(f_extentMatch);
            _addMetaEntry("extent_match", currentIdx++, _metaDict);

            //Personal prep match
            String prp_1 = "", prp_2 = "";
            for(Token t : m1.getTokenList())
                if(_prps.contains(t.toString().toLowerCase()))
                    prp_1 = t.toString().toLowerCase();
            for(Token t : m2.getTokenList())
                if(_prps.contains(t.toString().toLowerCase()))
                    prp_2 = t.toString().toLowerCase();
            int f_prpMatch = FALSE;
            if(!prp_1.isEmpty() && prp_1.equals(prp_2))
                f_prpMatch = TRUE;
            featureList.add(f_prpMatch);
            _addMetaEntry("prp_match", currentIdx++, _metaDict);

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
            _addMetaEntry("lex_type_match", currentIdx++, _metaDict);
            featureList.add(f_lexTypeMatch_other);
            _addMetaEntry("lex_type_match_other", currentIdx++, _metaDict);
            featureList.add(f_lexTypeMatch_only);
            _addMetaEntry("lex_type_match_only", currentIdx++, _metaDict);

            String cocoCat_1 = Mention.getLexicalEntry_cocoCategory(m1, true);
            String cocoCat_2 = Mention.getLexicalEntry_cocoCategory(m2, true);
            Double f_cocoCatMatch = 0.0;
            if(cocoCat_1 != null && cocoCat_2 != null){
                if(cocoCat_1.equals(cocoCat_2))
                    f_cocoCatMatch = 1.0;

                Set<String> lex_1 = new HashSet<>(Arrays.asList(cocoCat_1.split("/")));
                Set<String> lex_2 = new HashSet<>(Arrays.asList(cocoCat_2.split("/")));
                Set<String> intersection = new HashSet<>(lex_1);
                intersection.retainAll(lex_2);
                if(intersection.size() > 0)
                    f_cocoCatMatch = 0.5;
            }
            featureList.add(f_cocoCatMatch);
            _addMetaEntry("coco_cat_match", currentIdx++, _metaDict);

            //Chunk neighbor features -- left
            Chunk leftNeighbor_1 = _mentionChunkNeighborDict.get(m1)[0];
            Chunk leftNeighbor_2 = _mentionChunkNeighborDict.get(m2)[0];
            String leftChunkType_1 = null, leftChunkType_2 = null;
            if(leftNeighbor_1 != null)
                leftChunkType_1 = leftNeighbor_1.getChunkType();
            if(leftNeighbor_2 != null)
                leftChunkType_2 = leftNeighbor_2.getChunkType();
            Integer f_leftMatch = _getChunkTypeMatch(leftChunkType_1, leftChunkType_2);
            featureList.add(f_leftMatch);
            _addMetaEntry("left_chunk_match", currentIdx++, _metaDict);

            //Chunk neighbor features -- right
            Chunk rightNeighbor_1 = _mentionChunkNeighborDict.get(m1)[1];
            Chunk rightNeighbor_2 = _mentionChunkNeighborDict.get(m2)[1];
            String rightChunkType_1 = null, rightChunkType_2 = null;
            if(rightNeighbor_1 != null)
                rightChunkType_1 = rightNeighbor_1.getChunkType();
            if(rightNeighbor_2  != null)
                rightChunkType_2 = rightNeighbor_2.getChunkType();
            Integer f_rightMatch = _getChunkTypeMatch(rightChunkType_1, rightChunkType_2);
            featureList.add(f_rightMatch);
            _addMetaEntry("right_chunk_match", currentIdx++, _metaDict);

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
            _addMetaEntry("out_dep_match", currentIdx++, _metaDict);

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
            _addMetaEntry("det_plural_match", currentIdx++, _metaDict);

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
            _addMetaEntry("is_subj_match", currentIdx++, _metaDict);
            _addMetaEntry("is_obj_match", currentIdx++, _metaDict);
            featureList.add(f_subjOfMatch);
            featureList.add(f_objOfMatch);
            _addMetaEntry("subj_of_match", currentIdx++, _metaDict);
            _addMetaEntry("obj_of_match", currentIdx++, _metaDict);
            featureList.add(f_isSubj_1);
            featureList.add(f_isSubj_2);
            _addMetaEntry("is_subj_i", currentIdx++, _metaDict);
            _addMetaEntry("is_subj_j", currentIdx++, _metaDict);
            featureList.add(f_isObj_1);
            featureList.add(f_isObj_2);
            _addMetaEntry("is_obj_i", currentIdx++, _metaDict);
            _addMetaEntry("is_obj_j", currentIdx++, _metaDict);

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
            _addMetaEntry("semi_pronom_i", currentIdx++, _metaDict);
            _addMetaEntry("semi_pronom_j", currentIdx++, _metaDict);
            featureList.add(f_xOfY_1);
            featureList.add(f_xOfY_2);
            _addMetaEntry("x_of_y_i", currentIdx++, _metaDict);
            _addMetaEntry("x_of_y_j", currentIdx++, _metaDict);
            featureList.add(f_appos_1);
            featureList.add(f_appos_2);
            _addMetaEntry("appositive_i", currentIdx++, _metaDict);
            _addMetaEntry("appositive_j", currentIdx++, _metaDict);
            featureList.add(f_inList_1);
            featureList.add(f_inList_2);
            _addMetaEntry("in_list_i", currentIdx++, _metaDict);
            _addMetaEntry("in_list_j", currentIdx++, _metaDict);

            //neural features, meant to encode the rule based pronominal coref
            if(_forNeural) {
                int f_isAnimate_i = m1.getLexicalType().contains("people") ||
                        m1.getLexicalType().contains("animals") ? 1 : 0;
                int f_isAnimate_j = m2.getLexicalType().contains("people") ||
                        m2.getLexicalType().contains("animals") ? 1 : 0;
                featureList.add(f_isAnimate_i);
                featureList.add(f_isAnimate_j);
                _addMetaEntry("isAnimate_i", currentIdx++, _metaDict);
                _addMetaEntry("isAnimate_j", currentIdx++, _metaDict);
                int f_isThat_i = m1.toString().equalsIgnoreCase("that") ? 1 : 0;
                int f_isThat_j = m2.toString().equalsIgnoreCase("that") ? 1 : 0;
                featureList.add(f_isThat_i);
                featureList.add(f_isThat_j);
                _addMetaEntry("isThat_i", currentIdx++, _metaDict);
                _addMetaEntry("isThat_j", currentIdx++, _metaDict);

                int f_iToBej = FALSE;
                int f_iOfj = FALSE;
                if (m1.getCaptionIdx() == m2.getCaptionIdx() && m1.getIdx() < m2.getIdx()) {
                    List<Chunk> interstitialChunks =
                            _doc.getCaption(m1.getCaptionIdx()).getInterstitialChunks(m1, m2);
                    List<Token> interstitialTokens =
                            _doc.getCaption(m1.getCaptionIdx()).getInterstitialTokens(m1, m2);
                    if (interstitialChunks.size() == 1) {
                        String intrstlChunkType = interstitialChunks.get(0).getChunkType();
                        String intrstlChunkStr = interstitialChunks.get(0).toString().toLowerCase();
                        f_iToBej = intrstlChunkType.equals("VP") &&
                                (StringUtil.containsElement(Arrays.asList(_identityTerms), intrstlChunkStr) ||
                                        intrstlChunkStr.equals("is") || intrstlChunkStr.equals("are")) ? TRUE : FALSE;
                    }
                    if (interstitialTokens.size() == 1 &&
                            interstitialTokens.get(0).toString().equalsIgnoreCase("of")) {
                        f_iOfj = TRUE;
                    }
                }
                featureList.add(f_iToBej);
                featureList.add(f_iOfj);
                _addMetaEntry("iToBej", currentIdx++, _metaDict);
                _addMetaEntry("iOfj", currentIdx++, _metaDict);

                int f_isFirstInCap_i = m1.getIdx() == 0 ? TRUE : FALSE;
                int f_isFirstInCap_j = m2.getIdx() == 0 ? TRUE : FALSE;
                int f_adjacent_ij = m1.getCaptionIdx() == m2.getCaptionIdx() &&
                                    m1.getIdx() + 1 == m2.getIdx() ? TRUE : FALSE;
                featureList.add(f_isFirstInCap_i);
                featureList.add(f_isFirstInCap_j);
                featureList.add(f_adjacent_ij);
                _addMetaEntry("isFirstInCap_i", currentIdx++, _metaDict);
                _addMetaEntry("isFirstInCap_j", currentIdx++, _metaDict);
                _addMetaEntry("adjacent_ij", currentIdx++, _metaDict);
            }

            //new subset features
            int f_hasArticle_i = _articles.contains(m1.getTokenList().get(0).toString().toLowerCase()) ? TRUE : FALSE;
            int f_hasArticle_j = _articles.contains(m2.getTokenList().get(0).toString().toLowerCase()) ? TRUE : FALSE;
            featureList.add(f_hasArticle_i);
            featureList.add(f_hasArticle_j);
            _addMetaEntry("hasArticle_i", currentIdx++, _metaDict);
            _addMetaEntry("hasArticle_j", currentIdx++, _metaDict);
            int f_isMass_i = _masses.contains(m1.getTokenList().get(0).toString().toLowerCase()) ? TRUE : FALSE;
            int f_isMass_j = _masses.contains(m2.getTokenList().get(0).toString().toLowerCase()) ? TRUE : FALSE;
            featureList.add(f_isMass_i);
            featureList.add(f_isMass_j);
            _addMetaEntry("isMass_i", currentIdx++, _metaDict);
            _addMetaEntry("isMass_j", currentIdx++, _metaDict);
            int f_hasCollective_i = FALSE;
            int f_hasCollective_j = FALSE;
            for(Token t : m1.getTokenList())
                if(_collectives.contains(t.toString().toLowerCase()) || _collectives.contains(t.getLemma()))
                    f_hasCollective_i = TRUE;
            for(Token t : m2.getTokenList())
                if(_collectives.contains(t.toString().toLowerCase()) || _collectives.contains(t.getLemma()))
                    f_hasCollective_j = TRUE;
            featureList.add(f_hasCollective_i);
            featureList.add(f_hasCollective_j);
            _addMetaEntry("hasCollective_i", currentIdx++, _metaDict);
            _addMetaEntry("hasCollective_j", currentIdx++, _metaDict);
            int f_hasPortion_i = FALSE;
            int f_hasPortion_j = FALSE;
            for(Token t : m1.getTokenList())
                if(_portions.contains(t.toString().toLowerCase()) || _portions.contains(t.getLemma()))
                    f_hasPortion_i = TRUE;
            for(Token t : m2.getTokenList())
                if(_portions.contains(t.toString().toLowerCase()) || _portions.contains(t.getLemma()))
                    f_hasPortion_j = TRUE;
            featureList.add(f_hasPortion_i);
            featureList.add(f_hasPortion_j);
            _addMetaEntry("hasPortion_i", currentIdx++, _metaDict);
            _addMetaEntry("hasPortion_j", currentIdx++, _metaDict);
            int f_isSingular_i = m1.getHead().getPosTag().equals("NN") ||
                    m1.getHead().getPosTag().equals("NNP") ? TRUE : FALSE;
            int f_isSingular_j = m2.getHead().getPosTag().equals("NN") ||
                    m2.getHead().getPosTag().equals("NNP") ? TRUE : FALSE;
            int f_isPlural_i = m1.getHead().getPosTag().equals("NNS") ||
                    m1.getHead().getPosTag().equals("NNPS") ? TRUE : FALSE;
            int f_isPlural_j = m2.getHead().getPosTag().equals("NNS") ||
                    m2.getHead().getPosTag().equals("NNPS") ? TRUE : FALSE;
            featureList.add(f_isSingular_i);
            featureList.add(f_isSingular_j);
            featureList.add(f_isPlural_i);
            featureList.add(f_isPlural_j);
            _addMetaEntry("isSingular_i", currentIdx++, _metaDict);
            _addMetaEntry("isSingular_j", currentIdx++, _metaDict);
            _addMetaEntry("isPlural_i", currentIdx++, _metaDict);
            _addMetaEntry("isPlural_j", currentIdx++, _metaDict);
            int f_isSemi_i = m1.getPronounType() ==
                    Mention.PRONOUN_TYPE.SEMI ? TRUE : FALSE;
            int f_isSemi_j = m2.getPronounType() ==
                    Mention.PRONOUN_TYPE.SEMI ? TRUE : FALSE;
            featureList.add(f_isSemi_i);
            featureList.add(f_isSemi_j);
            _addMetaEntry("isSemi_i", currentIdx++, _metaDict);
            _addMetaEntry("isSemi_j", currentIdx++, _metaDict);
            int[] f_knownQuantity_i = new int[6], f_knownQuantity_j = new int[6];
            Arrays.fill(f_knownQuantity_i, FALSE);
            Arrays.fill(f_knownQuantity_j, FALSE);
            int knownQuantity_i = _getKnownQuantity(m1);
            int knownQuantity_j = _getKnownQuantity(m2);
            if(knownQuantity_i > 0 && knownQuantity_i <= 6)
                f_knownQuantity_i[knownQuantity_i-1] = TRUE;
            if(knownQuantity_j > 0 && knownQuantity_j <= 6)
                f_knownQuantity_j[knownQuantity_j-1] = TRUE;
            for(int i=1; i<=6; i++){
                featureList.add(f_knownQuantity_i[i-1]);
                featureList.add(f_knownQuantity_j[i-1]);
                _addMetaEntry("knownQuantity_i_" + i, currentIdx++, _metaDict);
                _addMetaEntry("knownQuantity_j_" + i, currentIdx++, _metaDict);
            }

            //Cardinality features
            if(!_forNeural && _includeCard){
                for(int i=0; i<12; i++){
                    featureList.add(_cardScores.get(m1.getUniqueID())[i]);
                    featureList.add(_cardScores.get(m2.getUniqueID())[i]);
                    _addMetaEntry("cardinality_i_" + i, currentIdx++, _metaDict);
                    _addMetaEntry("cardinality_j_" + i, currentIdx++, _metaDict);
                }
            }

            //Meta features; recall that head_not_lemma, while
            //providing symmetry, doesn't actually make sense
            Integer f_lemmaNotHead = f_lemmaMatch == TRUE && f_headMatch == FALSE ? TRUE : FALSE;
            featureList.add(f_lemmaNotHead);
            _addMetaEntry("lemma_not_head", currentIdx++, _metaDict);

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

            String distance_ij = "null";
            if(m1.getCaptionIdx() == m2.getCaptionIdx()){
                int dist = m2.getIdx() - m1.getIdx();
                if(0 < dist && dist <= 10)
                    distance_ij = String.valueOf(dist);
                else if(dist > 10)
                    distance_ij = ">10";
            }

            //Add one hot vectors, which internally adjust the feature vector but
            //doesn't adjust the idx
            if(!_forNeural){
                currentIdx = _addOneHotVector(headPair, fv, currentIdx,
                        _headPairs, "head_pair_onehot", _metaDict);
                currentIdx = _addOneHotVector(lemmaPair, fv, currentIdx,
                        _lemmaPairs, "lemma_pair_onehot", _metaDict);
                currentIdx = _addOneHotVector(subjOfPair, fv, currentIdx,
                        _subjOfPairs, "subj_of_onehot", _metaDict);
                currentIdx = _addOneHotVector(objOfPair, fv, currentIdx,
                        _objOfPairs, "obj_of_onehot", _metaDict);
                currentIdx = _addOneHotVector(firstWord_1, fv, currentIdx,
                        _dets, "det_1_onehot", _metaDict);
                currentIdx = _addOneHotVector(firstWord_2, fv, currentIdx,
                        _dets, "det_2_onehot", _metaDict);
                currentIdx = _addOneHotVector(numericPair, fv, currentIdx,
                        _numericPairs, "numeric_pair_onehot", _metaDict);
                currentIdx = _addOneHotVector(modPair, fv, currentIdx,
                        _modifierPairs, "modifier_pair_onehot", _metaDict);
                currentIdx = _addOneHotVector(leftPrepPair, fv, currentIdx,
                        _prepositionPairs, "left_preposition_pair_onehot", _metaDict);
                currentIdx = _addOneHotVector(rightPrepPair, fv, currentIdx,
                        _prepositionPairs, "right_preposition_pair_onehot", _metaDict);
                currentIdx = _addOneHotVector(distance_ij, fv, currentIdx,
                        _distances, "distance_ij", _metaDict);
            }
            currentIdx = _addOneHotVector(typePair, fv, currentIdx,
                    _typePairs, "lex_type_pair_onehot", _metaDict);
            currentIdx = _addOneHotVector(leftPair, fv, currentIdx,
                    _leftPairs, "left_pair_onehot", _metaDict);
            currentIdx = _addOneHotVector(rightPair, fv, currentIdx,
                    _rightPairs, "right_pair_onehot", _metaDict);
            currentIdx = _addOneHotVector(cocoCat_1 + "|" + cocoCat_2, fv, currentIdx,
                    _categoryPairs, "categoryPair_onehot", _metaDict);
            if(_forNeural){
                String pronomType_i = m1.getPronounType().toString();
                currentIdx = _addOneHotVector(pronomType_i, fv, currentIdx,
                        _pronounTypes, "pronoun_type_i_onehot", _metaDict);
                String pronomType_j = m1.getPronounType().toString();
                currentIdx = _addOneHotVector(pronomType_j, fv, currentIdx,
                        _pronounTypes, "pronoun_type_j_onehot", _metaDict);
            }

            //We treat hypernyms as -- not a onehot -- but a bag-of-words;
            //Given core concepts, we keep a vector where entry ij
            //is 1 if one of the mentions has concept i in its senses hypernyms
            //and the other has concept j
            //UPDATE: This never worked very well; toss if
            /*
            int start = currentIdx;
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
            _addMetaEntry("hypernym_bow", start, currentIdx, _metaDict);
            */

            //Finally, add a four-way label indicating if these mentions are
            //coreferent, subset, or null\

            /*
            Integer label = 0;
            if(!m1.getChainID().equals("0") && !m2.getChainID().equals("0")){
               String id_ij = Document.getMentionPairStr(m1, m2);
               String id_ji = Document.getMentionPairStr(m2, m1);
                if(m1.getChainID().equals(m2.getChainID()))
                    label = 1;
                else if(_subsetMentions.contains(id_ij))
                    label = 2;  //subset relation; assumes other
                                //vector from this pair will be 3
                else if(_subsetMentions.contains(id_ji))
                    label = 3; //Superset relation; assumes other
                               //vector from this pair will be 2
                else if(_partOfMentions.contains(id_ij))
                    label = 4; //Part of relations are asymmetrical
            }*/

            _addMetaEntry("max_idx", currentIdx+1, _metaDict);
            fv.label = _getLabel(m1, m2);
            fv.comments = Document.getMentionPairStr(m1, m2);
            return fv;
        }

        private static int _getChunkTypeMatch(String chunkType_1, String chunkType_2) {
            int typeMatch = UNK;
            if (chunkType_1 == null && chunkType_2 == null)
                typeMatch = TRUE;
            else if (chunkType_1 == null)
                typeMatch = FALSE;
            else if (chunkType_2 == null)
                typeMatch = FALSE;
            else
                typeMatch = chunkType_1.equals(chunkType_2) ? TRUE : FALSE;
            return typeMatch;
        }
    }
}
