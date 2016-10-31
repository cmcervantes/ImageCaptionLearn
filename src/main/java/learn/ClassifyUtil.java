package learn;

import core.Overlord;
import nlptools.Word2VecUtil;
import nlptools.WordnetUtil;
import structures.*;
import utilities.*;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.util.*;

public abstract class ClassifyUtil {
    private static final int UNK = 0;
    private static final int FALSE = 0;
    private static final int TRUE = 1;

    protected static String _outroot;

    /* Static collections used by subordinate threads */
    protected static Map<String, DoubleDict<String>> _imgLemmaCountDict;
    protected static Map<Mention, Chunk[]> _mentionChunkNeighborDict;
    protected static Map<String, String> _hypDict;
    protected static Set<Mention> _onlyTypeMentions;
    protected static Map<Mention, Chunk> _subjOfDict;
    protected static Map<Mention, Chunk> _objOfDict;
    protected static Word2VecUtil _w2vUtil;
    protected static Map<Mention, Chain> _mentionChainDict;
    protected static Set<String> _mentionPairsWithSubsetBoxes;
    private static Map<String, String> _clothAttrLex;
    protected static Set<Mention> _visualMentions;

    //lists from files
    private static Set<String> _colors;
    protected static Set<String> _detSet_singular;
    protected static Set<String> _detSet_plural;
    protected static List<String> _detList;
    protected static Set<String> _stopWords;

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
        _headList = getOneHotList(Overlord.resourcesDir + "hist_head.csv", 1);
        _typeList = getOneHotList(Overlord.resourcesDir + "hist_type.csv", 1000);
        _leftList = getOneHotList(Overlord.resourcesDir + "hist_left.csv", 1000);
        _rightList = getOneHotList(Overlord.resourcesDir + "hist_right.csv", 1000);
        _subjOfList = getOneHotList(Overlord.resourcesDir + "hist_subjOf.csv", 1);
        _objOfList = getOneHotList(Overlord.resourcesDir + "hist_objOf.csv", 1);
        _cardList = getOneHotList(Overlord.resourcesDir + "hist_cardinality.csv", 0);
        _pronounList = getOneHotList(Overlord.resourcesDir + "hist_pronoun.csv", 1);
        _pronounTypeList = getOneHotList(Overlord.resourcesDir + "hist_pronounType.csv", 0);

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

    public static void runAblation(String trainFile, String testFile,
                                   List<Integer> ignoredFeatIndices) {
        Logger.log("Reading feature vectors from file");
        List<FeatureVector> fvList = new ArrayList<>();
        List<String> trainVectorStrList = FileIO.readFile_lineList(trainFile);
        for (String trainVecStr : trainVectorStrList)
            fvList.add(FeatureVector.parseFeatureVector(trainVecStr));

        Logger.log("Ignored Indices: {" + StringUtil.listToString(ignoredFeatIndices, ", ") + "}");
        LiblinearSVM svm = new LiblinearSVM("L2R_LR_DUAL", 1.0, 0.001);
        svm.train_excludeIndices(fvList, ignoredFeatIndices);
        svm.evaluate(testFile);

        List<Integer> ablationIndices = new ArrayList<>();
        for (int i = 1; i < 30; i++)
            if (!ignoredFeatIndices.contains(i))
                ablationIndices.add(i);

        for (Integer ablationIdx : ablationIndices) {
            Logger.log("Ignoring " + ablationIdx);
            List<Integer> ignoredIndices = new ArrayList<>();
            ignoredIndices.add(ablationIdx);
            ignoredIndices.addAll(ignoredFeatIndices);
            svm.train_excludeIndices(fvList, ignoredIndices);
            svm.evaluate(testFile);
        }
        Logger.log("Ablation complete");
    }

    public static void trainSans(LiblinearSVM svm, String trainFile,
                                 Collection<Document> docSet, String sansType) {
        Logger.log("Retrieving IDs to exclude (%s)", sansType);
        Set<String> ignoredIDs = getSansIDs(docSet, sansType);

        Logger.log("Training (excluding %s pairs)", sansType);
        svm.train_excludeIDs(trainFile, ignoredIDs);
    }

    public static void evaluateSans(LiblinearSVM svm, String evalFile,
                                    Collection<Document> docSet, String sansType) {
        Logger.log("Retrieving IDs to exclude (%s)", sansType);
        Set<String> ignoredIDs = getSansIDs(docSet, sansType);

        Logger.log("Evaluating (excluding %s pairs)");
        svm.evaluate_excludeIDs(evalFile, ignoredIDs);
    }

    private static Set<String> getSansIDs(Collection<Document> docSet, String sansType) {
        Set<String> ignoreIDs = new HashSet<>();
        for (Document d : docSet) {
            for (int i = 0; i < d.getMentionList().size(); i++) {
                for (int j = i + 1; j < d.getMentionList().size(); j++) {
                    Mention m1 = d.getMentionList().get(i);
                    Mention m2 = d.getMentionList().get(j);

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

    public static void exportFeatures_nonvis(Collection<Document> docSet, String fileRoot) {
        exportFeatures_nonvis(docSet, fileRoot, true, -1);
    }

    public static void exportFeatures_nonvis(Collection<Document> docSet, String fileRoot,
                                             int numThreads) {
        exportFeatures_nonvis(docSet, fileRoot, false, numThreads);
    }

    private static void exportFeatures_nonvis(Collection<Document> docSet,
                                              String fileRoot, boolean useWord2Vec,
                                              int numThreads) {
        Logger.log("Feature Preprocessing: reading lists from files");
        //10k unique heads in train_excludeIndices; 61% of which occur more than once
        //17 unique types in train_excludeIndices; 70% of which occur more than 1k times
        //12 unique left chunk types in train_excludeIndices; 58% of which occur more than 1k times
        //10 unique right chunk types in train_excludeIndices; 70% of which occur more than 1k times
        //~4k unique subjOf verbs in train_excludeIndices; 61% of which occur more than once
        //~3k unique objOf verbs in train_excludeIndices; 61% of which occur more than once

        String filename = fileRoot + ".feats";
        if (useWord2Vec) {
            //if we're using word2vec, we don't need multithreading,
            //we're just getting pre-loaded embeddings
            Logger.log("Loading Word2Vec utility");
            Word2VecUtil w2vUtil = new Word2VecUtil(Overlord.word2vecPath, _headList);

            Logger.log("Opening [" + filename + "] for writing");
            BufferedWriter bw = null;
            try {
                bw = new BufferedWriter(new FileWriter(filename));
            } catch (IOException ioEx) {
                System.err.println("Could not save output file " + filename);
                System.exit(0);
            }

            Logger.log("Writing head word embeddings");
            for (Document d : docSet) {
                for (Mention m : d.getMentionList()) {
                    List<Double> embedding = w2vUtil.getVector(m.getHead().getText());
                    if (embedding == null) {
                        Double[] embeddingArr = new Double[300];
                        Arrays.fill(embeddingArr, 0.0);
                        embedding = Arrays.asList(embeddingArr);
                    }
                    double label = m.getChainID().equals("0") ? 1.0 : 0;
                    String comments = m.getUniqueID();
                    FeatureVector fv = new FeatureVector(embedding, label, comments);
                    try {
                        bw.write(fv.toString() + "\n");
                    } catch (IOException ioEx) {
                        System.err.println("Failed to save vector for: " + comments);
                        System.exit(0);
                    }
                }
            }

            Logger.log("Closing [" + filename + "]");
            try {
                bw.close();
            } catch (IOException ioEx) {
                System.err.println("Could not close output file " + filename);
            }
        } else {
            //exportFeatures(docSet, ExtractionThreadType.NONVIS, filename, numThreads);
        }
    }

    @Deprecated
    public static void exportFeatures_coref(Collection<Document> docSet,
                                            String fileRoot, boolean useWord2Vec,
                                            int numThreads) {
        Logger.log("Getting head->hypernym set dict");
        WordnetUtil wnUtil = new WordnetUtil(Overlord.wordnetDir);
        _hypDict = wnUtil.getHypernymDict(docSet);

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

        //open a word2vec clerk, using those heads
        _w2vUtil = null;
        if (useWord2Vec) {
            Logger.log("Loading vocabulary from Word2Vec");
            Set<String> vocab = new HashSet<>();
            for (Document d : docSet)
                for (Caption c : d.getCaptionList())
                    for (Token t : c.getTokenList())
                        vocab.add(t.getText().toLowerCase());
            _w2vUtil = new Word2VecUtil(Overlord.word2vecPath, vocab);
        } else {
            Logger.log("Word2vec feature computation disabled");
        }


        //for ultra-easy retrieval, associate mentions
        //with their parent chains
        Logger.log("Mapping mentions to visual chains");
        _mentionChainDict = new HashMap<>();
        for (Document d : docSet)
            for (Chain c : d.getChainSet())
                if (!c.getID().equals("0"))
                    for (Mention m : c.getMentionSet())
                        _mentionChainDict.put(m, c);

        //Create the lists necessary for our one-hot vectors
        //Our threshold for our tail of terms is 1000, below which
        // 0.3% of our pairs, in the typePair case
        // 0.1% of our pairs, in the leftPair case
        // 0.09% of our pairs, in the right pair case
        Logger.log("Reading one_hot lists");
        _typePairList = getOneHotList(Overlord.resourcesDir + "hist_typePair.csv", 1000);
        _leftPairList = getOneHotList(Overlord.resourcesDir + "hist_leftPair.csv", 1000);
        _rightPairList = getOneHotList(Overlord.resourcesDir + "hist_rightPair.csv", 1000);
        _headPairList = getOneHotList(Overlord.resourcesDir + "hist_headPair.csv", 1);
        _lemmaPairList = getOneHotList(Overlord.resourcesDir + "hist_lemmaPair.csv", 1);
        _subjOfPairList = getOneHotList(Overlord.resourcesDir + "hist_subjOfPair.csv", 1);
        _objOfPairList = getOneHotList(Overlord.resourcesDir + "hist_objOfPair.csv", 1);

        //get lemma counts for lemma_out feats
        Logger.log("Feature preprocessing (lemma counts)");
        _imgLemmaCountDict = new HashMap<>();
        for (Document d : docSet) {
            _imgLemmaCountDict.put(d.getID(), new DoubleDict<>());
            for (Mention m : d.getMentionList())
                _imgLemmaCountDict.get(d.getID()).increment(m.getHead().getLemma().toLowerCase().trim());
        }

        //get the chunks to the left and right of the mention
        //UPDATE: Imagine we're at chunk 2 in "[NP a boy] [VP rides] [NP a sled]"
        //        Our chunk neighbors should be VP/END, since the second NP is bookended
        //        by a VP and the end of the caption, respectively.
        //        Consider adding a full stop at the end, such that the caption is
        //        "[NP a boy] [VP rides] [NP a sled] [NULL .]"
        //        Because the way captions / tokens / chunks interact, our chunk neighbors
        //        will be VP/NULL, but _should_ be VP/END. So we'll need to handle this
        //        Keep in mind, however, that if our caption looks like
        //        "[NP a boy] and [NP a girl] [VP sled] [NULL .]"
        //        The chunk neighbors for the first chunk will be START/NULL, since simple
        //        conjunctions are not chunks
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

        //exportFeatures(docSet, ExtractionThreadType.COREF, fileRoot + ".feats", numThreads);
    }

    @Deprecated
    public static void exportFeatures_subset(Collection<Document> docSet,
                                             String fileRoot, int numThreads) {
        Logger.log("Getting head->hypernym set dict");
        WordnetUtil wnUtil = new WordnetUtil(Overlord.wordnetDir);
        _hypDict = wnUtil.getHypernymDict(docSet);

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

        Logger.log("Getting proper subset labels");
        _mentionPairsWithSubsetBoxes = new HashSet<>();
        for (Document d : docSet) {
            for (Caption c : d.getCaptionList()) {
                for (int i = 0; i < c.getMentionList().size(); i++) {
                    Mention m1 = c.getMentionList().get(i);
                    Set<BoundingBox> boxes_1 = d.getBoxSetForMention(m1);
                    for (int j = i + 1; j < c.getMentionList().size(); j++) {
                        Mention m2 = c.getMentionList().get(j);
                        Set<BoundingBox> boxes_2 = d.getBoxSetForMention(m2);
                        Set<BoundingBox> boxIntersect = new HashSet<>(boxes_1);
                        boxIntersect.retainAll(boxes_2);

                        Set<String> types1 = new HashSet<>(Arrays.asList(m1.getLexicalType().split("/")));
                        Set<String> types2 = new HashSet<>(Arrays.asList(m1.getLexicalType().split("/")));
                        Set<String> typeIntersect = new HashSet<>(types1);
                        typeIntersect.retainAll(types2);
                        boolean sameType = !typeIntersect.isEmpty();
                        boolean isPronom1 = m1.getPronounType() != Mention.PRONOUN_TYPE.NONE;
                        boolean isPronom2 = m2.getPronounType() != Mention.PRONOUN_TYPE.NONE;
                        boolean isAgent1 = m1.getLexicalType().contains("people") || m1.getLexicalType().contains("animals");
                        boolean isAgent2 = m2.getLexicalType().contains("people") || m2.getLexicalType().contains("animals");
                        boolean bothHaveBoxes = !boxes_1.isEmpty() && !boxes_2.isEmpty();

                        //Subset mentions must be _either_ of matching types _or_ are pronouns
                        //NOTE: they can't _both_ be pronouns
                        if (bothHaveBoxes && (isAgent1 || isAgent2)) {
                            if (boxes_2.containsAll(boxes_1) && boxes_2.size() > boxIntersect.size()) {
                                _mentionPairsWithSubsetBoxes.add(Document.getMentionPairStr(m1, m2, true, true));
                            } else if (boxes_1.containsAll(boxes_2) && boxes_1.size() > boxIntersect.size()) {
                                _mentionPairsWithSubsetBoxes.add(Document.getMentionPairStr(m2, m1, true, true));
                            }
                        }
                    }
                }
            }
        }

        //exportFeatures(docSet, ExtractionThreadType.SUBSET, fileRoot + ".feats", numThreads);
    }

    public static void exportFeatures_pairwise(Collection<Document> docSet, String outroot, int numThreads) {
        _outroot = outroot + ".feats";

        Logger.log("Initializing lists");
        ClassifyUtil.initLists();

        Logger.log("Getting head->hypernym set dict");
        WordnetUtil wnUtil = new WordnetUtil(Overlord.wordnetDir);
        _hypDict = wnUtil.getHypernymDict(docSet);

        Logger.log("Mapping mentions to visual chains");
        _visualMentions = new HashSet<>();
        for (Document d : docSet)
            for (Mention m : d.getMentionList())
                if (!m.getChainID().equals("0"))
                    _visualMentions.add(m);

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

/*
        Logger.log("Feature preprocessing (loading word2vec)");
        Set<String> vocab = new HashSet<>();
        Set<String> stopWords = new HashSet<>();
        stopWords.addAll(_detList);
        stopWords.addAll(_pronounList);
        stopWords.add(","); stopWords.add("."); stopWords.add("!");
        stopWords.add(";"); stopWords.add(":"); stopWords.add("?");
        for(Document d : docSet)
            for(Mention m : d.getMentionList())
                for(Token t : m.getTokenList())
                    if(!stopWords.contains(t.getText()))
                        vocab.add(t.getText().toLowerCase().trim());
        _w2vUtil = new Word2VecUtil(Overlord.word2vecPath, vocab);*/
        exportFeatures(docSet, ExtractionThreadType.PAIRWISE, numThreads);
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
                    String word = StringUtil.keepAlpha(t.getText()).toLowerCase();
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
    public static Map<String, Set<Mention[]>> pronominalCoref(Collection<Document> docSet) {
        Map<String, Set<Mention[]>> imgMentionPairDict = new HashMap<>();
        WordnetUtil wnUtil = new WordnetUtil(Overlord.wordnetDir);
        Map<String, String> hypDict = wnUtil.getHypernymDict(docSet);

        for (Document d : docSet) {
            for (Caption c : d.getCaptionList()) {
                boolean DEBUG_DELETEME = false;
                if (c.toString().equals("A man with jeans , black sneakers , a belt , and dreadlocks is jumping and creating a shadow on the wall next to him .")) {
                    DEBUG_DELETEME = true;
                }

                //start at the second mention, since first mentions cannot have valid antecedents
                for (int i = 1; i < c.getMentionList().size(); i++) {
                    Mention m_pronom = c.getMentionList().get(i);
                    Mention.PRONOUN_TYPE pronounType = m_pronom.getPronounType();

                    if (pronounType != Mention.PRONOUN_TYPE.NONE &&
                            pronounType != Mention.PRONOUN_TYPE.INDEFINITE) {

                        if (DEBUG_DELETEME)
                            System.out.println("pro: " + m_pronom.toString() + "\t" + m_pronom.toDebugString());

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
                            if (DEBUG_DELETEME)
                                System.out.println("ant: " + m_ante.toString() + "\t" + m_ante.toDebugString());

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
                                boolean genderMatch = getGenderMatch(m_pronom, m_ante, hypDict);
                                if (genderMatch) {
                                    candAnteList.add(m_ante);

                                    if (DEBUG_DELETEME)
                                        System.out.println("added");

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
                            if (DEBUG_DELETEME)
                                System.out.printf("Cands: %d; Type: %s", candAnteList.size(), pronounType.toString());

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
                                                !candAnte.getCardinality().isAmbiguous()) {
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
                            if (!c1.isAmbiguous() && !c2.isAmbiguous()) {
                                if (c1.getValue() < c2.getValue())
                                    supSub = new Mention[]{m2, m1};
                                else if (c2.getValue() < c1.getValue())
                                    supSub = new Mention[]{m1, m2};
                            } //2 sub 1
                            else if (c1.isAmbiguous() && !c2.isAmbiguous()) {
                                supSub = new Mention[]{m1, m2};
                            } //1 sub 2
                            else if (!c1.isAmbiguous() && c2.isAmbiguous()) {
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

    @Deprecated
    private static List<Mention> _retainCorefAnte(Mention pronom, List<Mention> anteList) {
        List<Mention> corefAnteList = new ArrayList<>();
        for (Mention m : anteList) {
            Boolean pluralPronom =
                    Mention.PRONOUN_TYPE.getIsPlural(pronom.toString().toLowerCase());
            if (pluralPronom != null) {
                if ((pluralPronom && m.getCardinality().getValue() > 1) ||
                        (!pluralPronom && m.getCardinality().getValue() == 1)) {
                    if (getGenderMatch(pronom, m, _hypDict))
                        corefAnteList.add(m);
                }
            }
        }
        return corefAnteList;
    }

    /**
     * We want to assign coreference based on pronoun type,
     * according to the below heuristics
     * <p>
     * 1) Subjective / Objective / Special ("it")
     * - Attach to furthest antecedent; usually main subject
     * 2) Reflexive / Reciprocal
     * - Attach to the nearest antecedent; usually in same clause
     * 3) Relative / Special ("that")
     * - If adjacent left neighbor is candidate, attach (NP [who/that])
     * - If XofY construction, exclude X, attach to nearest
     * - Otherwise attach nearest antecedent
     * 4) Indefinite
     * - No attachment
     * 5) Other
     * - "another", "other", "others"; no attachment
     * - "both"; attach to a animate, 2-card antecedent
     * - "one"; no attachment
     *
     * @param docSet
     * @return
     */
    public static Map<String, Map<String, Set<Mention[]>>> pronomRelation(Collection<Document> docSet) {
        Map<String, Map<String, Set<Mention[]>>> imgRelDict = new HashMap<>();
        WordnetUtil wnUtil = new WordnetUtil(Overlord.wordnetDir);
        _hypDict = wnUtil.getHypernymDict(docSet);

        for (Document d : docSet) {
            Map<String, Set<Mention[]>> relDict = new HashMap<>();
            for (Caption c : d.getCaptionList()) {
                //start at the second mention, since we're only concerned with
                //pronominal mentions with non-pronominal antecedents
                for (int i = 1; i < c.getMentionList().size(); i++) {
                    Mention m_pronom = c.getMentionList().get(i);
                    Mention.PRONOUN_TYPE pronounType = m_pronom.getPronounType();

                    //both subset and coreference relations only occur between
                    //non-indefinite pronouns
                    if (pronounType != Mention.PRONOUN_TYPE.NONE &&
                            pronounType != Mention.PRONOUN_TYPE.INDEFINITE) {

                        //collect left-to-right candidate antecedents
                        List<Mention> candAnteList = new ArrayList<>();
                        for (int j = i - 1; j >= 0; j--) {
                            Mention m_ante = c.getMentionList().get(j);
                            if (m_ante.getPronounType() == Mention.PRONOUN_TYPE.NONE)
                                candAnteList.add(m_ante);
                        }
                        int lastIdx = candAnteList.size() - 1;
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
                        List<Mention> candAnteList_coref = _retainCorefAnte(m_pronom, candAnteList);
                        if (!candAnteList_coref.isEmpty()) {
                         /* 1) Subjective / Objective */
                            if (pronounType == Mention.PRONOUN_TYPE.SUBJECTIVE_SINGULAR ||
                                    pronounType == Mention.PRONOUN_TYPE.SUBJECTIVE_PLURAL ||
                                    pronounType == Mention.PRONOUN_TYPE.OBJECTIVE_SINGULAR ||
                                    pronounType == Mention.PRONOUN_TYPE.OBJECTIVE_PLURAL ||
                                    m_pronom.toString().toLowerCase().equals("it")) {
                                pair = new Mention[]{candAnteList_coref.get(lastIdx), m_pronom};
                            } /* 2) Reflexive */ else if (pronounType == Mention.PRONOUN_TYPE.REFLEXIVE_SINGULAR ||
                                    pronounType == Mention.PRONOUN_TYPE.REFLEXIVE_PLURAL ||
                                    pronounType == Mention.PRONOUN_TYPE.RECIPROCAL) {
                                pair = new Mention[]{candAnteList_coref.get(0), m_pronom};
                            } /* 3) Relative */ else if (pronounType == Mention.PRONOUN_TYPE.RELATIVE ||
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
                                        if (candAnte.getCardinality().getValue() == 2 &&
                                                !candAnte.getCardinality().isAmbiguous()) {
                                            pair = new Mention[]{candAnteList_coref.get(k), m_pronom};
                                            break;
                                        }
                                    }
                                }
                            }
                        }

                        /**If the pronoun isn't attached to a coreferent
                         * antecedent, detect a subset relation
                         */
                        if (pair == null) {
                            boolean pronomIsAnimate =
                                    m_pronom.getPronounType().isAnimate();

                        }


                    }
                }
            }
        }

        return null;
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
        WordnetUtil wnUtil = new WordnetUtil(Overlord.wordnetDir);
        Map<String, String> hypDict = wnUtil.getHypernymDict(docSet);
        Map<Mention, String> genderDict = new HashMap<>();
        for (Document d : docSet) {
            for (Mention m : d.getMentionList()) {
                Set<String> hypSet = new HashSet<>();
                getHypernymSet(m.getHead().getLemma(), hypDict, hypSet);
                String gender = m.getGender(hypSet);
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
            text = t.getText();
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
            String tokenText = t.getText().toLowerCase();
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

    /**
     * Exports the joint box/mention embedding used in the affinity model, resulting
     * in a collection of [4096 + 300] vectors, for the img + txt feats
     *
     * @param docSet
     * @param outroot
     * @param numThreads
     */
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
                    String text = t.getText().toLowerCase().trim();
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

    /**Returns the set of hypernyms in the dict associated with this lemma
     * or any of its hypernyms
     *
     * @param lemma
     * @param hypDict
     * @return
     */
    private static void getHypernymSet(String lemma,
              Map<String, String> hypDict, Set<String> hypSet)
    {
        if(!hypDict.containsKey(lemma)){
            String hypLemma = hypDict.get(lemma);
            if(!hypSet.contains(hypLemma)){
                hypSet.add(hypLemma);
                getHypernymSet(hypLemma, hypDict, hypSet);
            }
        }
    }


    public static boolean getGenderMatch(Mention m1, Mention m2, Map<String, String> hypDict)
    {
        Set<String> hypSet_1 = new HashSet<>();
        ClassifyUtil.getHypernymSet(m1.getHead().getLemma(), hypDict, hypSet_1);
        String gender_1 = m1.getGender(hypSet_1);
        Set<String> hypSet_2 = new HashSet<>();
        ClassifyUtil.getHypernymSet(m2.getHead().getLemma(), hypDict, hypSet_2);
        String gender_2 = m2.getGender(hypSet_2);
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
        COREF, NONVIS, SUBSET, AFFINITY, NULL, PAIRWISE
    }

    private static class ExtractionThread extends Thread
    {
        private Document _doc;
        private ExtractionThreadType _type;
        public Collection<FeatureVector> fvSet;

        public ExtractionThread(Document doc, ExtractionThreadType type)
        {
            _doc = doc;
            _type = type;
            fvSet = new HashSet<>();
        }

        public void run()
        {
            switch(_type){
                case COREF: run_coref();
                    break;
                case NONVIS: run_nonvis();
                    break;
                case SUBSET: run_subset();
                    break;
                case AFFINITY: run_affinity();
                    break;
                case PAIRWISE: run_pairwise();
                    break;
            }
        }

        private void run_coref()
        {
            for(int i=0; i<_doc.getMentionList().size(); i++){
                Mention m1 = _doc.getMentionList().get(i);
                if(m1.getPronounType() != Mention.PRONOUN_TYPE.NONE)
                    continue;

                for(int j=i+1; j<_doc.getMentionList().size(); j++){
                    Mention m2 = _doc.getMentionList().get(j);

                    if(m2.getPronounType() != Mention.PRONOUN_TYPE.NONE)
                        continue;

                    //load our mentions and their heads / lemmas
                    String head_1 = m1.getHead().getText().toLowerCase();
                    String head_2 = m2.getHead().getText().toLowerCase();
                    String lemma_1 = m1.getHead().getLemma().toLowerCase();
                    String lemma_2 = m2.getHead().getLemma().toLowerCase();

                    //get the headPair and lemmaPair feats
                    String headPair = StringUtil.getAlphabetizedPair(head_1, head_2).toLowerCase();
                    headPair = headPair.replace(",", "");
                    String lemmaPair = StringUtil.getAlphabetizedPair(lemma_1, lemma_2).toLowerCase();
                    lemmaPair = lemmaPair.replace(",", "");

                    //compute headMatch and lemmaMatch feats
                    Integer headMatch = head_1.equals(head_2) ? TRUE : FALSE;
                    Integer lemmaMatch = lemma_1.equals(lemma_2) ? TRUE : FALSE;

                    //compute mention text feats
                    String extent_1 = m1.toString().replace(head_1, "").toLowerCase().trim();
                    String extent_2 = m2.toString().replace(head_2, "").toLowerCase().trim();
                    Integer extentMatch = UNK;
                    if(!extent_1.isEmpty() || !extent_2.isEmpty())
                        extentMatch = extent_1.equalsIgnoreCase(extent_2) ? TRUE : FALSE;

                    Integer substring = lemma_1.contains(lemma_2) ||
                            lemma_2.contains(lemma_1) ? TRUE : FALSE;

                    //if either lemma count is over one (two if they match),
                    //we know that lemma occurs somewhere other than the
                    //original context of the pair
                    //We should trip whenever we see lemma_1 in caption_1 twice
                    //or - if lemma_1==lemma_2 and caption_1==caption_2 - three times
                    //UPDATE: like only_type, this is true only when they match _and_
                    //        are both the only instance of their lemma in their caption
                    Integer lemmaOutMatch = FALSE;
                    if(_imgLemmaCountDict.containsKey(_doc.getID())){
                        int thresh = 2;
                        if(m1.getCaptionIdx() == m2.getCaptionIdx())
                            thresh++;
                        if(lemmaMatch == TRUE){
                            if(_imgLemmaCountDict.get(_doc.getID()).get(lemma_1) < thresh &&
                                    _imgLemmaCountDict.get(_doc.getID()).get(lemma_2) < thresh){
                                lemmaOutMatch = TRUE;
                            }
                        }
                    }

                    //get chunk neighbor features
                    Chunk leftNeighbor_1 = _mentionChunkNeighborDict.get(m1)[0];
                    Chunk rightNeighbor_1 = _mentionChunkNeighborDict.get(m1)[1];
                    Chunk leftNeighbor_2 = _mentionChunkNeighborDict.get(m2)[0];
                    Chunk rightNeighbor_2 = _mentionChunkNeighborDict.get(m2)[1];
                    String leftChunkType_1 = null;
                    String rightChunkType_1 = null;
                    String leftChunkType_2 = null;
                    String rightChunkType_2 = null;
                    if(leftNeighbor_1 != null)
                        leftChunkType_1 = leftNeighbor_1.getChunkType();
                    if(rightNeighbor_1 != null)
                        rightChunkType_1 = rightNeighbor_1.getChunkType();
                    if(leftNeighbor_2 != null)
                        leftChunkType_2 = leftNeighbor_2.getChunkType();
                    if(rightNeighbor_2  != null)
                        rightChunkType_2 = rightNeighbor_2.getChunkType();
                    String leftPair = StringUtil.getAlphabetizedPair(leftChunkType_1, leftChunkType_2);
                    String rightPair = StringUtil.getAlphabetizedPair(rightChunkType_1, rightChunkType_2);
                    Integer leftMatch = getChunkTypeMatch(leftChunkType_1, leftChunkType_2);
                    Integer rightMatch = getChunkTypeMatch(rightChunkType_1, rightChunkType_2);

                    //get our hypernym sets for these two lemmas
                    Set<String> hypernymSet_1 = new HashSet<>();
                    getHypernymSet(lemma_1.toLowerCase(), _hypDict, hypernymSet_1);
                    Set<String> hypernymSet_2 = new HashSet<>();
                    getHypernymSet(lemma_2.toLowerCase(), _hypDict, hypernymSet_2);

                    Integer hypernymMatch = UNK;
                    if(hypernymSet_1 != null && hypernymSet_2 != null &&
                            !hypernymSet_1.isEmpty() && !hypernymSet_2.isEmpty()) {
                        //we have a match if the intersection has any elements
                        hypernymSet_1.retainAll(hypernymSet_2);
                        if(hypernymSet_1.size() > 0)
                            hypernymMatch = TRUE;
                        else
                            hypernymMatch = FALSE;
                    }

                    //compute type match feats
                    String type_1 = m1.getLexicalType();
                    String type_2 = m2.getLexicalType();
                    String typeType = null;
                    Double typeMatch = (double)UNK;
                    Double otherTypeMatch = (double)FALSE;
                    if(type_1 != null && type_2 != null) {
                        typeMatch = Mention.getLexicalTypeMatch(m1, m2);
                        if(typeMatch == 0.0)
                            typeMatch = (double)FALSE;

                        typeType = StringUtil.getAlphabetizedPair(type_1, type_2);

                        if(type_1.equals("other") && type_2.equals("other"))
                            otherTypeMatch = (double)TRUE;
                        else if(type_1.contains("other") && type_2.contains("other"))
                            otherTypeMatch = 0.5;
                    }

                    Cardinality c1 = m1.getCardinality();
                    Cardinality c2 = m2.getCardinality();
                    Integer headPosMatch = UNK;
                    Integer pluralMatch = UNK;
                    Token headToken_1 = m1.getHead();
                    Token headToken_2 = m2.getHead();
                    if(headToken_1 != null && headToken_2 != null) {
                        headPosMatch = headToken_1.getPosTag().equals(headToken_2.getPosTag()) ? 1 : 0;
                        if(c1 == null || c2 == null){
                            pluralMatch = TRUE;
                        } else if(c1.getValue() == 1 && !c1.isAmbiguous() &&
                                c2.getValue() == 1 && !c2.isAmbiguous()){
                            pluralMatch = TRUE;
                        } else if(c1.getValue() > 1 && c2.getValue() > 1){
                            pluralMatch = TRUE;
                        } else {
                            pluralMatch = FALSE;
                        }
                    }

                    Integer onlyTypeMatch = FALSE;
                    if(m1.getLexicalType().equals(m2.getLexicalType())){
                        if(_onlyTypeMentions.contains(m1) == _onlyTypeMentions.contains(m2))
                            onlyTypeMatch = TRUE;
                    }

                    //our cardinality matches are
                    //  -1: mismatch
                    //   0: unk
                    //  .5: ambiguous match
                    //   1: unambiguous match
                    Double cardMatch = (double)UNK;
                    if(c1 != null && c2 != null){
                        if(Cardinality.approxEqual(c1,c2)){
                            if(c1.isAmbiguous() || c2.isAmbiguous())
                                cardMatch = .5;
                            else
                                cardMatch = (double)TRUE;
                        } else {
                            cardMatch = (double)FALSE;
                        }
                    }

                    //compute the meta feats
                    Integer headNotLemma =
                            headMatch == TRUE && lemmaMatch == FALSE ? TRUE : FALSE;
                    Integer lemmaNotHead =
                            headMatch == FALSE && lemmaMatch == TRUE ? TRUE : FALSE;

                    //compute dep type feats
                    DependencyNode root_1 = _doc.getCaption(m1.getCaptionIdx()).getRootNode();
                    DependencyNode root_2 = _doc.getCaption(m2.getCaptionIdx()).getRootNode();
                    Integer depMatch = UNK;
                    if(root_1 != null && root_2 != null){
                        Set<String> outRel_1 = root_1.getOutRelations(m1);
                        Set<String> outRel_2 = root_2.getOutRelations(m2);
                        Set<String> outRel = new HashSet<>(outRel_1);
                        outRel.retainAll(outRel_2);
                        depMatch = outRel.isEmpty() ? FALSE : TRUE;
                    }

                    //compute the determiner plural match for these mentions
                    //UPDATE: assume the determiner will be the first token
                    Integer detPluralMatch = UNK;
                    String firstWord_1 = m1.getTokenList().get(0).getText().toLowerCase();
                    String firstWord_2 = m2.getTokenList().get(0).getText().toLowerCase();
                    boolean firstWordIsSingular_1 = _detSet_singular.contains(firstWord_1);
                    boolean firstWordIsSingular_2 = _detSet_singular.contains(firstWord_2);
                    boolean firstWordIsPlural_1 = _detSet_plural.contains(firstWord_1);
                    boolean firstWordIsPlural_2 = _detSet_plural.contains(firstWord_2);
                    if((firstWordIsSingular_1 && firstWordIsSingular_2) ||
                            (firstWordIsPlural_1 && firstWordIsPlural_2)) {
                        detPluralMatch = TRUE;
                    } else if((firstWordIsSingular_1 && firstWordIsPlural_2) ||
                            (firstWordIsPlural_1 && firstWordIsSingular_2)) {
                        //we only explicitly declare the match as false
                        //if we _know_ the other is part of the other set.
                        detPluralMatch = FALSE;
                    }

                    //get our verb features
                    Chunk subjOf_1 = _subjOfDict.get(m1);
                    Chunk subjOf_2 = _subjOfDict.get(m2);
                    Chunk objOf_1 = _objOfDict.get(m1);
                    Chunk objOf_2 = _objOfDict.get(m2);
                    String subjOfStr_1 = null;
                    if(subjOf_1 != null)
                        subjOfStr_1 = subjOf_1.getTokenList().get(subjOf_1.getTokenList().size()-1).getText().toLowerCase();
                    String subjOfStr_2 = null;
                    if(subjOf_2 != null)
                        subjOfStr_2 = subjOf_2.getTokenList().get(subjOf_2.getTokenList().size()-1).getText().toLowerCase();
                    String objOfStr_1 = null;
                    if(objOf_1 != null)
                        objOfStr_1 = objOf_1.getTokenList().get(objOf_1.getTokenList().size()-1).getText().toLowerCase();
                    String objOfStr_2 = null;
                    if(objOf_2 != null)
                        objOfStr_2 = objOf_2.getTokenList().get(objOf_2.getTokenList().size()-1).getText().toLowerCase();


                    Integer isSubjMatch = FALSE;
                    Integer isObjMatch = FALSE;
                    if(subjOf_1 != null && subjOf_2 != null)
                        isSubjMatch = TRUE;
                    if(objOf_1 != null && objOf_2 != null)
                        isObjMatch = TRUE;

                    //mark our matches as UNK if we either don't have any verb _or_
                    //if we didn't find an entry for this mention at all
                    //(which... shouldn't occur)
                    Integer subjectOfMatch = UNK;
                    Integer objectOfMatch = UNK;
                    if(subjOfStr_1 != null)
                        subjectOfMatch = subjOfStr_1.equals(subjOfStr_2) ? TRUE : FALSE;
                    else if(subjOfStr_2 != null)
                        subjectOfMatch = FALSE; //reaching here is always a non-match
                    if(objOfStr_1 != null)
                        subjectOfMatch = objOfStr_1.equals(objOfStr_2) ? TRUE : FALSE;
                    else if(objOfStr_2 != null)
                        objectOfMatch = FALSE; //reaching here is always a non-match

                    //concatenate and alphabetize our verb lemmas
                    String subjOfPair =
                            StringUtil.getAlphabetizedPair(subjOfStr_1, subjOfStr_2);
                    String objOfPair =
                            StringUtil.getAlphabetizedPair(objOfStr_1, objOfStr_2);

                    //get our cosine similarities
                    double lemmaCosineSim = 0;
                    double subjOfCosineSim = 0;
                    double objOfCosineSim = 0;
                    double headCosineSim = 0;
                    double mentionCosineSim_add = 0;
                    double mentionCosineSim_mean = 0;
                    double mentionCosineSim_mult = 0;
                    if(_w2vUtil != null){
                        lemmaCosineSim = _w2vUtil.getWord2VecSim(lemma_1, lemma_2);
                        subjOfCosineSim = _w2vUtil.getWord2VecSim(subjOfStr_1, subjOfStr_2);
                        objOfCosineSim = _w2vUtil.getWord2VecSim(objOfStr_1, objOfStr_2);
                        headCosineSim = _w2vUtil.getWord2VecSim(head_1, head_2);

                        mentionCosineSim_add =
                                Util.cosineSimilarity(getMentionVector("add", m1),
                                        getMentionVector("add", m2));
                        mentionCosineSim_mean =
                                Util.cosineSimilarity(getMentionVector("mean", m1),
                                        getMentionVector("mean", m2));
                        mentionCosineSim_mult =
                                Util.cosineSimilarity(getMentionVector("mult", m1),
                                        getMentionVector("mult", m2));

                    }

                    //add our simple features into a vector
                    FeatureVector featVector = new FeatureVector();
                    Object[] simpleFeats = {lemmaMatch, extentMatch, typeMatch,
                            pluralMatch, substring, hypernymMatch,
                            detPluralMatch, headPosMatch, headMatch,
                            lemmaOutMatch, headNotLemma, lemmaNotHead,
                            isSubjMatch, isObjMatch, depMatch, onlyTypeMatch,
                            cardMatch, leftMatch, rightMatch, subjectOfMatch,
                            objectOfMatch, otherTypeMatch};
                    for(int k=0; k<simpleFeats.length; k++){
                        Object f = simpleFeats[k];
                        Double val = Double.parseDouble(f.toString());
                        if(val != 0)
                            featVector.addFeature(k+1, val);
                    }
                    int currentIdx = 1 + simpleFeats.length;

                    //if we've enabled word2vec features, add them
                    if(_w2vUtil != null){
                        featVector.addFeature(currentIdx, lemmaCosineSim);
                        currentIdx++;
                        featVector.addFeature(currentIdx, headCosineSim);
                        currentIdx++;
                        featVector.addFeature(currentIdx, subjOfCosineSim);
                        currentIdx++;
                        featVector.addFeature(currentIdx, objOfCosineSim);
                        currentIdx++;
                        featVector.addFeature(currentIdx, mentionCosineSim_add);
                        currentIdx++;
                        featVector.addFeature(currentIdx, mentionCosineSim_mean);
                        currentIdx++;
                        featVector.addFeature(currentIdx, mentionCosineSim_mult);
                        currentIdx++;
                    }

                    //add our one-hots
                    if(_typePairList.indexOf(typeType) > -1)
                        featVector.addFeature(currentIdx + _typePairList.indexOf(typeType) + 1, 1.0);
                    currentIdx += 1 + _typePairList.size();
                    if(_leftPairList.indexOf(leftPair) > -1)
                        featVector.addFeature(currentIdx + _leftPairList.indexOf(leftPair) + 1, 1.0);
                    currentIdx += 1 + _leftPairList.size();
                    if(_rightPairList.indexOf(rightPair) > -1)
                        featVector.addFeature(currentIdx + _rightPairList.indexOf(rightPair) + 1, 1.0);
                    currentIdx += 1 + _rightPairList.size();
                    if(_lemmaPairList.indexOf(lemmaPair) > -1)
                        featVector.addFeature(currentIdx + _lemmaPairList.indexOf(lemmaPair) + 1, 1.0);
                    currentIdx += 1 + _lemmaPairList.size();
                    if(_headPairList.indexOf(headPair) > -1)
                        featVector.addFeature(currentIdx + _headPairList.indexOf(headPair) + 1, 1.0);
                    currentIdx += 1 + _headPairList.size();
                    if(_subjOfPairList.indexOf(subjOfPair) > -1)
                        featVector.addFeature(currentIdx + _subjOfPairList.indexOf(subjOfPair)+1, 1.0);
                    currentIdx += 1 + _subjOfPairList.size();
                    if(_objOfPairList.indexOf(objOfPair) > -1)
                        featVector.addFeature(currentIdx + _objOfPairList.indexOf(objOfPair)+1, 1.0);
                    currentIdx += 1 + _objOfPairList.size();

                    //finally, add the pair ID and the label and write the
                    //vector to the file
                    Integer label = 0;
                    if(_mentionChainDict.containsKey(m1) && _mentionChainDict.containsKey(m2))
                        label = _mentionChainDict.get(m1).equals(_mentionChainDict.get(m2)) ? 1 : 0;
                    featVector.comments = Document.getMentionPairStr(m1,m2,true);
                    featVector.label = label;
                    fvSet.add(featVector);
                }
            }
        }

        private void run_nonvis()
        {
            for(Caption c : _doc.getCaptionList()){
                for(Mention m : c.getMentionList()){
                    //ignore pronominal mentions
                    if(m.getPronounType() != Mention.PRONOUN_TYPE.NONE)
                        continue;

                    String head = m.getHead().getText().toLowerCase();
                    String type = m.getLexicalType().toLowerCase();
                    Chunk subjOf = c.getSubjectOf(m);
                    String subjOfStr = null;
                    if(subjOf != null)
                        subjOfStr = c.getTokenList().get(subjOf.getTokenRange()[1]).getText().toLowerCase();
                    Chunk objOf = c.getObjectOf(m);
                    String objOfStr = null;
                    if(objOf != null)
                        objOfStr = c.getTokenList().get(objOf.getTokenRange()[1]).getText().toLowerCase();

                    int leftChunkIdx = -1;
                    int rightChunkIdx = Integer.MAX_VALUE;
                    if(!m.getChunkList().isEmpty()){
                        leftChunkIdx = m.getChunkList().get(0).getIdx() - 1;
                        rightChunkIdx = m.getChunkList().get(m.getChunkList().size()-1).getIdx() + 1;
                    }
                    String left = "START";
                    if(leftChunkIdx >= 0)
                        left = c.getChunkList().get(leftChunkIdx).getChunkType();
                    String right = "END";
                    if(rightChunkIdx < c.getChunkList().size())
                        right = c.getChunkList().get(rightChunkIdx).getChunkType();

                    //for now, each mention is just a combination of these
                    //one hots
                    FeatureVector fv = new FeatureVector();
                    int currentIdx = 1;
                    if(_typeList.indexOf(type) > -1)
                        fv.addFeature(currentIdx + _typeList.indexOf(type) + 1, 1.0);
                    currentIdx += 1 + _typeList.size();
                    if(_leftList.indexOf(left) > -1)
                        fv.addFeature(currentIdx + _leftList.indexOf(left) + 1, 1.0);
                    currentIdx += 1 + _leftList.size();
                    if(_rightList.indexOf(right) > -1)
                        fv.addFeature(currentIdx + _rightList.indexOf(right) + 1, 1.0);
                    currentIdx += 1 + _rightList.size();
                    if(_headList.indexOf(head) > -1)
                        fv.addFeature(currentIdx + _headList.indexOf(head) + 1, 1.0);
                    currentIdx += 1 + _headList.size();
                    if(_subjOfList.indexOf(subjOfStr) > -1)
                        fv.addFeature(currentIdx + _subjOfList.indexOf(subjOfStr)+1, 1.0);
                    currentIdx += 1 + _subjOfList.size();
                    if(_objOfList.indexOf(objOfStr) > -1)
                        fv.addFeature(currentIdx + _objOfList.indexOf(objOfStr)+1, 1.0);
                    currentIdx += 1 + _objOfList.size();
                    fv.label = m.getChainID().equals("0") ? 1 : 0;
                    fv.comments = m.getUniqueID();
                    fvSet.add(fv);
                }
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
                    List<Double> vecToAdd = _w2vUtil.getVector(t.getText().toLowerCase().trim());
                    if(vecToAdd.size() < 300){
                        System.err.printf("Token %s has vector size %d\n",
                                t.getText(), vecToAdd.size());
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
                    fv_mention.forEach(v -> concat.add(v));

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

        private void run_subset()
        {
            for(Caption c : _doc.getCaptionList()){
                for(int i=0; i<c.getMentionList().size(); i++){
                    Mention m1 = c.getMentionList().get(i);
                    for(int j=i+1; j<c.getMentionList().size(); j++){
                        Mention m2 = c.getMentionList().get(j);
                        fvSet.add(getSubsetFeatureVector(c, m1, m2));
                        fvSet.add(getSubsetFeatureVector(c, m2, m1));
                    }
                }
            }

        }

        private void run_pairwise()
        {
            for(int i=0; i<_doc.getMentionList().size(); i++){
                Mention m1 = _doc.getMentionList().get(i);
                if(m1.getPronounType() != Mention.PRONOUN_TYPE.NONE)
                    continue;

                for(int j=i+1; j<_doc.getMentionList().size(); j++){
                    Mention m2 = _doc.getMentionList().get(j);
                    if(m2.getPronounType() != Mention.PRONOUN_TYPE.NONE)
                        continue;

                    fvSet.add(getPairwiseFeatureVector(m1, m2));
                    fvSet.add(getPairwiseFeatureVector(m2, m1));
                }
            }
        }

        /**Returns a subset feature vector for m1 and m2;
         * computes features as if m1 is a subset of m2
         *
         * @param m1
         * @param m2
         * @return
         */
        private FeatureVector getSubsetFeatureVector(Caption c, Mention m1, Mention m2)
        {
            FeatureVector fv = new FeatureVector();
            String normHead1 = m1.getHead().getText().toLowerCase();
            String normHead2 = m2.getHead().getText().toLowerCase();

            int currentIdx = 1;

            //Cardinality comparison;
            //  if a < b; 1
            //  elif b is ambiguous; 0.5
            //  else -1
            Cardinality c1 = m1.getCardinality();
            Cardinality c2 = m2.getCardinality();
            if(!c1.isAmbiguous() && !c2.isAmbiguous() && c1.getValue() < c2.getValue()){
                fv.addFeature(currentIdx, TRUE);
            } else if(c2.isAmbiguous()){
                fv.addFeature(currentIdx, 0.5);
            } else {
                fv.addFeature(currentIdx, FALSE);
            }
            currentIdx++;

            //is plural a/b
            if(c1.getValue() > 1)
                fv.addFeature(currentIdx, TRUE);
            else
                fv.addFeature(currentIdx, FALSE);
            currentIdx++;
            if(c2.getValue() > 1)
                fv.addFeature(currentIdx, TRUE);
            else
                fv.addFeature(currentIdx, FALSE);
            currentIdx++;

            //Simple lemma match
            if(m1.getHead().getLemma().equalsIgnoreCase(m2.getHead().getLemma()))
                fv.addFeature(currentIdx, TRUE);
            else
                fv.addFeature(currentIdx, FALSE);
            currentIdx++;

            //a is "one"
            if(normHead1.equals("one"))
                fv.addFeature(currentIdx, TRUE);
            else
                fv.addFeature(currentIdx, FALSE);
            currentIdx++;

            //second is they/them
            if(normHead2.equals("they") || normHead2.equals("them"))
                fv.addFeature(currentIdx, TRUE);
            else
                fv.addFeature(currentIdx, FALSE);
            currentIdx++;

            //type match
            if(m1.getLexicalType().equals(m2.getLexicalType())){
                fv.addFeature(currentIdx, (double)TRUE);
            } else {
                Set<String> types1 = new HashSet<>(Arrays.asList(m1.getLexicalType().split("/")));
                Set<String> types2 = new HashSet<>(Arrays.asList(m2.getLexicalType().split("/")));
                Set<String> typeIntersect = new HashSet<>(types1);
                typeIntersect.retainAll(types2);
                if(!typeIntersect.isEmpty())
                    fv.addFeature(currentIdx, 0.5);
                else
                    fv.addFeature(currentIdx, FALSE);
            }
            currentIdx++;

            //simple gender match
            Set<String> hypSet1 = new HashSet<>();
            Set<String> hypSet2 = new HashSet<>();
            getHypernymSet(m1.getHead().getLemma().toLowerCase(), _hypDict, hypSet1);
            getHypernymSet(m2.getHead().getLemma().toLowerCase(), _hypDict, hypSet2);
            String g1 = m1.getGender(hypSet1);
            String g2 = m2.getGender(hypSet2);
            if(g1.equals(g2) && !g1.equals("neuter"))
                fv.addFeature(currentIdx, TRUE);
            else if(g1.equals("neuter") || g2.equals("neuter"))
                fv.addFeature(currentIdx, 0.5);
            else
                fv.addFeature(currentIdx, FALSE);
            currentIdx++;

                        /*
                        //determine if - assuming one exists in this caption - these two
                        //mentions are in an aside relation
                        boolean inAsideRel = false;
                        if(hasAside){
                            if(c.getTokenList().get(m1.getTokenRange()[1]+1).getText().equals(",")){
                                //The first mention is our necessary prefix ("X , ..."); Determine
                                //if the second mention precedes the next comma
                                int tokIdx = m1.getTokenRange()[1]+2;
                                boolean foundComma = false;
                                while(tokIdx < c.getTokenList().size() && !foundComma){
                                    Token t = c.getTokenList().get(tokIdx);
                                    if(t.getText().equals(","))
                                        foundComma = true;
                                    else if(t.mentionIdx == m2.getIdx())
                                        inAsideRel = true;
                                    tokIdx++;
                                }
                            }
                        }
                        if(inAsideRel)
                            fv.addFeature(currentIdx, TRUE);
                        else
                            fv.addFeature(currentIdx, FALSE);
                        currentIdx++;

                        //determine if these two mentions are part of a single
                        //list construction
                        boolean inSameConstr = false;
                        for(List<Mention> mentionConstr : mentionConstrList)
                            if(mentionConstr.contains(m1) && mentionConstr.contains(m2))
                                inSameConstr = true;
                        if(inSameConstr)
                            fv.addFeature(currentIdx, TRUE);
                        else
                            fv.addFeature(currentIdx, FALSE);
                        currentIdx++;
                        */

            //first is subject; second is subject; same for object
            if(c.getSubjectOf(m1) != null)
                fv.addFeature(currentIdx, TRUE);
            else
                fv.addFeature(currentIdx, FALSE);
            currentIdx++;
            if(c.getSubjectOf(m2) != null)
                fv.addFeature(currentIdx, TRUE);
            else
                fv.addFeature(currentIdx, FALSE);
            currentIdx++;
            if(c.getObjectOf(m1) != null)
                fv.addFeature(currentIdx, TRUE);
            else
                fv.addFeature(currentIdx, FALSE);
            currentIdx++;
            if(c.getObjectOf(m2) != null)
                fv.addFeature(currentIdx, TRUE);
            else
                fv.addFeature(currentIdx, FALSE);
            currentIdx++;

            //represent the presence of a determiner prefix as onehots
            String firstWord1 = m1.getTokenList().get(0).getText().toLowerCase();
            String firstWord2 = m2.getTokenList().get(0).getText().toLowerCase();
            if(_detList.indexOf(firstWord1) > -1)
                fv.addFeature(currentIdx + _detList.indexOf(firstWord1) + 1, 1.0);
            currentIdx += 1 + _detList.size();
            if(_detList.indexOf(firstWord2) > -1)
                fv.addFeature(currentIdx + _detList.indexOf(firstWord2) + 1, 1.0);
            currentIdx += 1 + _detList.size();

            Chunk leftNeighbor_1 = _mentionChunkNeighborDict.get(m1)[0];
            Chunk rightNeighbor_1 = _mentionChunkNeighborDict.get(m1)[1];
            Chunk leftNeighbor_2 = _mentionChunkNeighborDict.get(m2)[0];
            Chunk rightNeighbor_2 = _mentionChunkNeighborDict.get(m2)[1];
            String leftChunkType_1 = null;
            String rightChunkType_1 = null;
            String leftChunkType_2 = null;
            String rightChunkType_2 = null;
            if(leftNeighbor_1 != null)
                leftChunkType_1 = leftNeighbor_1.getChunkType();
            if(rightNeighbor_1 != null)
                rightChunkType_1 = rightNeighbor_1.getChunkType();
            if(leftNeighbor_2 != null)
                leftChunkType_2 = leftNeighbor_2.getChunkType();
            if(rightNeighbor_2  != null)
                rightChunkType_2 = rightNeighbor_2.getChunkType();
            String leftPair = leftChunkType_1 + "|" + leftChunkType_2;
            String rightPair = rightChunkType_1 + "|" + rightChunkType_2;

            Chunk subjOf_1 = _subjOfDict.get(m1);
            Chunk subjOf_2 = _subjOfDict.get(m2);
            Chunk objOf_1 = _objOfDict.get(m1);
            Chunk objOf_2 = _objOfDict.get(m2);
            String subjOfStr_1 = null;
            if(subjOf_1 != null)
                subjOfStr_1 = subjOf_1.getTokenList().get(subjOf_1.getTokenList().size()-1).getText().toLowerCase();
            String subjOfStr_2 = null;
            if(subjOf_2 != null)
                subjOfStr_2 = subjOf_2.getTokenList().get(subjOf_2.getTokenList().size()-1).getText().toLowerCase();
            String objOfStr_1 = null;
            if(objOf_1 != null)
                objOfStr_1 = objOf_1.getTokenList().get(objOf_1.getTokenList().size()-1).getText().toLowerCase();
            String objOfStr_2 = null;
            if(objOf_2 != null)
                objOfStr_2 = objOf_2.getTokenList().get(objOf_2.getTokenList().size()-1).getText().toLowerCase();
            String subjOfPair = subjOfStr_1 + "|" + subjOfStr_2;
            String objOfPair = objOfStr_1 + "|" + objOfStr_2;

            String headPair = normHead1 + "|" + normHead2;
            String lemmaPair = m1.getHead().getLemma() + "|" + m2.getHead().getLemma();
            lemmaPair = lemmaPair.toLowerCase();
            String typePair = m1.getLexicalType() + m2.getLexicalType();

            if(_headPairList.indexOf(headPair) > -1)
                fv.addFeature(currentIdx + _headPairList.indexOf(headPair) + 1, 1.0);
            currentIdx += 1 + _headPairList.size();
            if(_lemmaPairList.indexOf(lemmaPair) > -1)
                fv.addFeature(currentIdx + _lemmaPairList.indexOf(lemmaPair) + 1, 1.0);
            currentIdx += 1 + _lemmaPairList.size();
            if(_subjOfPairList.indexOf(subjOfPair) > -1)
                fv.addFeature(currentIdx + _subjOfPairList.indexOf(subjOfPair)+1, 1.0);
            currentIdx += 1 + _subjOfPairList.size();
            if(_objOfPairList.indexOf(objOfPair) > -1)
                fv.addFeature(currentIdx + _objOfPairList.indexOf(objOfPair)+1, 1.0);
            currentIdx += 1 + _objOfPairList.size();
            if(_typePairList.indexOf(typePair) > -1)
                fv.addFeature(currentIdx + _typePairList.indexOf(typePair) + 1, 1.0);
            currentIdx += 1 + _typePairList.size();
            if(_leftPairList.indexOf(leftPair) > -1)
                fv.addFeature(currentIdx + _leftPairList.indexOf(leftPair) + 1, 1.0);
            currentIdx += 1 + _leftPairList.size();
            if(_rightPairList.indexOf(rightPair) > -1)
                fv.addFeature(currentIdx + _rightPairList.indexOf(rightPair) + 1, 1.0);
            currentIdx += 1 + _rightPairList.size();

            //store the vectors
            String pairID = Document.getMentionPairStr(m1, m2, true, true);
            fv.comments = pairID;
            fv.label = _mentionPairsWithSubsetBoxes.contains(pairID) ? 1 : 0;

            return fv;
        }

        /**Return a complete pairwise feature vector
         *
         * @param m1
         * @param m2
         * @return
         */
        private FeatureVector getPairwiseFeatureVector(Mention m1, Mention m2)
        {
            List<Object> featureList = new ArrayList<>();

            //Head matches
            String head_1 = m1.getHead().getText().toLowerCase();
            String head_2 = m2.getHead().getText().toLowerCase();
            Integer f_headMatch = head_1.equals(head_2) ? TRUE : FALSE;
            Integer f_headPOSMatch = m1.getHead().getPosTag().equals(m2.getHead().getPosTag()) ? TRUE : FALSE;
            featureList.add(f_headMatch);
            featureList.add(f_headPOSMatch);

            //Lemma match / substring feat
            String lemma_1 = m1.getHead().getLemma().toLowerCase();
            String lemma_2 = m2.getHead().getLemma().toLowerCase();
            Integer f_lemmaMatch = lemma_1.equals(lemma_2) ? TRUE : FALSE;
            Integer f_substring = lemma_1.contains(lemma_2) ||
                    lemma_2.contains(lemma_1) ? TRUE : FALSE;
            featureList.add(f_lemmaMatch);
            featureList.add(f_substring);

            //Indicate whether the lemmas match _and_ are the only
            //instance of their lemma in their caption
            /*
            Integer f_onlyLemmaMatch = FALSE;
            if(_imgLemmaCountDict.containsKey(_doc.getID())) {
                if(f_lemmaMatch == TRUE) {
                    boolean onlyLemma_1 = false;
                    boolean onlyLemma_2 = false;



                    for(Mention m : _doc.getCaption(m1.getCaptionIdx()).getMentionList()){
                        if(!m.equals(m1))
                    }


                    for(Mention m : c.getMentionList()) {
                        if(!m.equals(m1) && !m.equals(m2)) {
                            if(m.getHead().getLemma().equals(m1.getHead().getLemma())) {
                                onlyLemma_1 = true;
                            } else if(m.getHead().getLemma().equals(m2.getHead().getLemma())) {
                                onlyLemma_2 = true;
                            }
                        }
                    }

                    if(onlyLemma_1 && onlyLemma_2)
                        f_onlyLemmaMatch = TRUE;
                }
            }
            featureList.add(f_onlyLemmaMatch);*/

            //Extent match
            String extent_1 = m1.toString().replace(m1.getHead().getText(), "").toLowerCase();
            String extent_2 = m2.toString().replace(m2.getHead().getText(), "").toLowerCase();
            Integer f_extentMatch = UNK;
            if(!extent_1.isEmpty() || !extent_2.isEmpty())
                f_extentMatch = extent_1.equalsIgnoreCase(extent_2) ? TRUE : FALSE;
            featureList.add(f_extentMatch);

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
            featureList.add(f_lexTypeMatch_other);
            featureList.add(f_lexTypeMatch_only);

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

            //Hypernym features
            Set<String> hypernymSet_1 = new HashSet<>();
            getHypernymSet(lemma_1.toLowerCase(), _hypDict, hypernymSet_1);
            Set<String> hypernymSet_2 = new HashSet<>();
            getHypernymSet(lemma_2.toLowerCase(), _hypDict, hypernymSet_2);
            Integer f_hypMatch = UNK;
            Double f_genderMatch = (double)UNK;
            if(hypernymSet_1 != null && hypernymSet_2 != null &&
                    !hypernymSet_1.isEmpty() && !hypernymSet_2.isEmpty()) {
                //we have a match if the intersection has any elements
                hypernymSet_1.retainAll(hypernymSet_2);
                if(hypernymSet_1.size() > 0)
                    f_hypMatch = TRUE;
                else
                    f_hypMatch = FALSE;

                String g1 = m1.getGender(hypernymSet_1);
                String g2 = m2.getGender(hypernymSet_2);
                if(g1.equals(g2) && !g1.equals("neuter"))
                    f_genderMatch = (double)TRUE;
                else if(g1.equals("neuter") || g2.equals("neuter"))
                    f_genderMatch = 0.5;
                else
                    f_genderMatch = (double)FALSE;
            }
            featureList.add(f_hypMatch);
            featureList.add(f_genderMatch);

            //Cardinality features
            Cardinality c1 = m1.getCardinality();
            Cardinality c2 = m2.getCardinality();
            Double f_cardMatch = (double)UNK;
            Double f_cardLess = (double)UNK;
            Integer f_pluralMatch = UNK;
            Integer f_singPLural = UNK;
            Integer f_isPlural_1 = UNK;
            Integer f_isPlural_2 = UNK;
            if(c1 != null && c2 != null) {
                if(Cardinality.approxEqual(c1,c2)) {
                    if(c1.isAmbiguous() || c2.isAmbiguous())
                        f_cardMatch = 0.5;
                    else
                        f_cardMatch = (double)TRUE;
                } else {
                    f_cardMatch = (double)FALSE;
                }

                //Cardinality less feature: 1 if a < b;
                // 0.5 is b is ambig (and thus relationship _could_ be less);
                // else -1
                if(c1.getValue() < c2.getValue() && !c1.isAmbiguous() && !c2.isAmbiguous()) {
                    f_cardLess = (double) TRUE;
                } else if(c2.isAmbiguous()){
                    f_cardLess = 0.5;
                } else {
                    f_cardLess = (double)FALSE;
                }

                //plurality match refers to whether both cardinalities
                //refer to a single element or multiple elements
                boolean c1_sing = c1.getValue() == 1 && !c1.isAmbiguous();
                boolean c2_sing = c2.getValue() == 1 && !c2.isAmbiguous();
                if(c1_sing && c2_sing)
                    f_pluralMatch = TRUE;
                else if(c1.getValue() > 1 && c2.getValue() > 1)
                    f_pluralMatch = TRUE;
                else
                    f_pluralMatch = FALSE;

                f_isPlural_1 = c1_sing ? FALSE : TRUE;
                f_isPlural_2 = c2_sing ? FALSE : TRUE;

                //singPlural reflects whether m1 is singular and m2 is plural
                if(c1_sing && !c2_sing)
                    f_singPLural = TRUE;
                else
                    f_singPLural = FALSE;
            }
            featureList.add(f_cardMatch);
            featureList.add(f_cardLess);
            featureList.add(f_pluralMatch);
            featureList.add(f_singPLural);
            featureList.add(f_isPlural_1);
            featureList.add(f_isPlural_2);

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

            //Determiner plural match (assume the first word is
            //the determiner candidate); FALSE is only assigned when
            //both have determiners of different pluralities
            String firstWord_1 = m1.getTokenList().get(0).getText().toLowerCase();
            String firstWord_2 = m2.getTokenList().get(0).getText().toLowerCase();
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

            //Verb features
            Chunk subjOf_1 = _subjOfDict.get(m1), subjOf_2 = _subjOfDict.get(m2);
            Chunk objOf_1 = _objOfDict.get(m1), objOf_2 = _objOfDict.get(m2);
            String subjOfStr_1 = subjOf_1 == null ? null :
                    subjOf_1.getTokenList().get(subjOf_1.getTokenList().size()-1).getText().toLowerCase();
            String subjOfStr_2 = subjOf_2 == null ? null :
                    subjOf_2.getTokenList().get(subjOf_2.getTokenList().size()-1).getText().toLowerCase();
            String objOfStr_1 = objOf_1 == null ? null :
                    objOf_1.getTokenList().get(objOf_1.getTokenList().size()-1).getText().toLowerCase();
            String objOfStr_2 = objOf_2 == null ? null :
                    objOf_2.getTokenList().get(objOf_2.getTokenList().size()-1).getText().toLowerCase();
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
            featureList.add(f_subjOfMatch);
            featureList.add(f_objOfMatch);
            featureList.add(f_isSubj_1);
            featureList.add(f_isSubj_2);
            featureList.add(f_isObj_1);
            featureList.add(f_isObj_2);

            //Meta features
            Integer f_headNotLemma = f_headMatch == TRUE && f_lemmaMatch == FALSE ? TRUE : FALSE;
            Integer f_lemmaNotHead = f_lemmaMatch == TRUE && f_headMatch == FALSE ? TRUE : FALSE;
            featureList.add(f_headNotLemma);
            featureList.add(f_lemmaNotHead);

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

            //Add one hot vectors, which internally adjust the feature vector but
            //doesn't adjust the idx (so we'll need to return it instead)
            int currentIdx = featureList.size() + 1;
            currentIdx = addOneHotVector(fv, currentIdx, _headPairList, headPair);
            currentIdx = addOneHotVector(fv, currentIdx, _lemmaPairList, lemmaPair);
            currentIdx = addOneHotVector(fv, currentIdx, _typePairList, typePair);
            currentIdx = addOneHotVector(fv, currentIdx, _leftPairList, leftPair);
            currentIdx = addOneHotVector(fv, currentIdx, _rightPairList, rightPair);
            currentIdx = addOneHotVector(fv, currentIdx, _subjOfList, subjOfPair);
            currentIdx = addOneHotVector(fv, currentIdx, _objOfList, objOfPair);
            currentIdx = addOneHotVector(fv, currentIdx, _detList, firstWord_1);
            currentIdx = addOneHotVector(fv, currentIdx, _detList, firstWord_2);

            //Finally, add a three-way label indicating if these mentions are
            //coreferent, subset, or null\
            Integer label = 0;
            if(_visualMentions.contains(m1) && _visualMentions.contains(m2)){
                if(m1.getChainID().equals(m2.getChainID()))
                    label = 1;
                else if(_doc.getBoxesAreSubset(m1,m2))
                    label = 2;  //subset relation; assumes other
                                //vector from this pair will be 3
                else if(_doc.getBoxesAreSubset(m2, m1))
                    label = 3; //Superset relation; assumes other
                               //vector from this pair will be 2
            }

            fv.label = label;
            fv.comments = Document.getMentionPairStr(m1, m2, true, true);
            return fv;
        }

        private FeatureVector _getPairwiseFeatureVector(Mention m1, Mention m2)
        {
            double[] emptyArr = new double[300];
            Arrays.fill(emptyArr, 0.0);
            List<Double> emptyVec = new ArrayList<>();
            for(double val : emptyArr)
                emptyVec.add(val);

            List<List<Double>> vectorList_1 = new ArrayList<>();
            for(Token t : m1.getTokenList()){
                String text = t.getText().toLowerCase().trim();
                List<Double> vector = _w2vUtil.getVector(text);
                if(StatisticalUtil.getSum(vector) == 0)
                    vectorList_1.add(vector);
            }
            if(vectorList_1.isEmpty())
                vectorList_1.add(new ArrayList<>(emptyVec));
            List<Double> x1 = Util.vectorMean(vectorList_1);
            List<List<Double>> vectorList_2 = new ArrayList<>();
            for(Token t : m2.getTokenList()){
                String text = t.getText().toLowerCase().trim();
                List<Double> vector = _w2vUtil.getVector(text);
                if(StatisticalUtil.getSum(vector) == 0)
                    vectorList_2.add(vector);
            }
            if(vectorList_2.isEmpty())
                vectorList_2.add(new ArrayList<>(emptyVec));
            List<Double> x2 = Util.vectorMean(vectorList_2);
            List<Double> x = new ArrayList<>();
            x.addAll(x1);
            x.addAll(x2);

            Double f_lexTypeMatch = (double)UNK;
            if(m1.getLexicalType() != null && m2.getLexicalType() != null) {
                f_lexTypeMatch = Mention.getLexicalTypeMatch(m1, m2);
                if(f_lexTypeMatch == 0.0)
                    f_lexTypeMatch = (double)FALSE;
            }

            //Finally, add a three-way label indicating if these mentions are
            //coreferent, subset, or null
            Integer label = 0;
            if(_visualMentions.contains(m1) && _visualMentions.contains(m2)){
                if(m1.getChainID().equals(m2.getChainID()))
                    label = 1;
                else if(_doc.getBoxesAreSubset(m1,m2))
                    label = 2;
            }

            return new FeatureVector(x, label, Document.getMentionPairStr(m1, m2, true, true));
        }

        private int addOneHotVector(FeatureVector fv, int currentIdx,
                                    List<String> oneHotList, String item)
        {
            if(oneHotList.indexOf(item) > -1)
                fv.addFeature(currentIdx + oneHotList.indexOf(item) + 1, 1.0);
            return currentIdx + oneHotList.size() + 1;
        }
    }
}
