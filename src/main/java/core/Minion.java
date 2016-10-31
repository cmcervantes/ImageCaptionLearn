package core;

import de.bwaldvogel.liblinear.*;
import learn.BinaryClassifierScoreDict;
import learn.ClassifyUtil;
import learn.FeatureVector;
import out.OutTable;
import statistical.ScoreDict;
import structures.*;
import utilities.*;

import java.io.*;
import java.util.*;

public class Minion
{
    public static void export_subsetHeterogType(Collection<Document> docSet)
    {
        DoubleDict<String> linkDict = new DoubleDict<>();

        DoubleDict<String> heterogHeads = new DoubleDict<>();
        OutTable ot = new OutTable("cap_id", "m1", "m1_idx", "m1_head", "m1_type",
                "m2", "m2_idx", "m2_head", "m2_type", "case", "cap1", "cap2");
        for(Document d : docSet){
            for(int i=0; i<d.getMentionList().size(); i++){
                Mention m1 = d.getMentionList().get(i);
                if(m1.getPronounType() != Mention.PRONOUN_TYPE.NONE)
                    continue;

                for(int j=i+1; j<d.getMentionList().size(); j++){
                    Mention m2 = d.getMentionList().get(j);
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
        for(Document d : docSet){
            int numMentions = d.getMentionList().size();
            for(int i=0; i<numMentions; i++){
                Mention m1 = d.getMentionList().get(i);
                int numTokens_1 = m1.getTokenList().size();
                for(int j=i+1; j<numMentions; j++){
                    Mention m2 = d.getMentionList().get(j);
                    int numTokens_2 = m2.getTokenList().size();

                    String penult_1 = null, penult_2 = null;
                    if(numTokens_1 > 1)
                        penult_1 = m1.getTokenList().get(numTokens_1-2).getText().toLowerCase();
                    if(numTokens_2 > 1)
                        penult_2 = m2.getTokenList().get(numTokens_2-2).getText().toLowerCase();
                    String ult_1 = m1.getTokenList().get(numTokens_1-1).getText().toLowerCase();
                    String ult_2 = m2.getTokenList().get(numTokens_2-1).getText().toLowerCase();

                    if(penult_1 != null && penult_1.equals(ult_2) ||
                       penult_2 != null && penult_2.equals(ult_1)){
                        Caption c1 = d.getCaption(m1.getCaptionIdx());
                        Caption c2 = d.getCaption(m2.getCaptionIdx());
                        ot.addRow(d.getID(), m1.toString(), m2.toString(), c1.toString(), c2.toString());
                    }
                }
            }
        }
        ot.writeToCsv("ex_penultFilter", true);
    }


    public static void buildImageCaptionDB(String corefFile, String releaseDir, String dbName)
    {
        Logger.log("Loading documents from coref file");
        Collection<Document> docSet_coref = DocumentLoader.getDocumentSet(
                corefFile,
                Overlord.lexPath, Overlord.resourcesDir);
        Map<String, Document> docDict_coref = new HashMap<>();
        docSet_coref.forEach(d -> docDict_coref.put(d.getID(), d));

        Set<String> typeSet = new HashSet<>();
        for(Document d : docSet_coref)
            for(Mention m : d.getMentionList())
                typeSet.add(m.getLexicalType());
        System.out.println(StringUtil.listToString(typeSet, " | "));

        Logger.log("Loading documents from flickr30kEntities file");
        Collection<Document> docSet_flickr =
                DocumentLoader.getDocumentSet(releaseDir,
                                              Overlord.resourcesDir);
        Map<String, Document> docDict_flickr = new HashMap<>();
        docSet_flickr.forEach(d -> docDict_flickr.put(d.getID(), d));
        typeSet = new HashSet<>();
        for(Document d : docDict_flickr.values())
            for(Mention m : d.getMentionList())
                typeSet.add(m.getLexicalType());
        System.out.println(StringUtil.listToString(typeSet, " | "));

        Logger.log("Merging documents");
        for(String docID : docDict_coref.keySet())
            docDict_coref.get(docID).loadBoxesFromDocument(docDict_flickr.get(docID));

        Logger.log("Uploading everything to the DB");
        DBConnector conn = new DBConnector(dbName);
        try{
            DocumentLoader.populateDocumentDB(conn, docDict_coref.values(), 100000, 1);
        } catch(Exception ex){
            utilities.Logger.log(ex);
        }

        /*
        Map<String, Integer> imgMap = legacy.ImgTable.getImgDict();
        Set<String> reviewedImgSet = legacy.ImgTable.getReviewedImgSet();
        Set<Object[]> paramSet = new HashSet<>();
        for(String imgID : imgMap.keySet()){
            int reviewed = reviewedImgSet.contains(imgID) ? 1 : 0;
            paramSet.add(new Object[]{imgMap.get(imgID), reviewed, imgID + ".jpg"});
        }
        String query = "UPDATE image SET cross_val=?, reviewed=? WHERE img_id=?;";
        try{
            conn.update(query, paramSet);
        }catch(Exception ex){
            Logger.log(ex);
        }*/
    }

    public static void exportPronomAttrEx(Collection<Document> docSet)
    {
        Map<String, Set<Mention[]>> pronomCoref =
                ClassifyUtil.pronominalCoref(docSet);
        OutTable ot_pronom = new OutTable("cap_id", "pronom_m",
                "pronom_idx", "pronom_start_token",
                "pronom_end_token", "ante_m", "ante_idx",
                "ante_start_token", "ante_end_token");
        for(String docID : pronomCoref.keySet()){
            for(Mention[] mPair: pronomCoref.get(docID)){
                Mention pronom = mPair[1];
                Mention ante = mPair[0];
                ot_pronom.addRow(docID + "#" + pronom.getCaptionIdx(),
                        pronom.toString().replace(",", ""), pronom.getIdx(),
                        pronom.getTokenRange()[0], pronom.getTokenRange()[1],
                        ante.toString().replace(",", ""), ante.getIdx(),
                        ante.getTokenRange()[0], ante.getTokenRange()[1]);
            }
        }
        ot_pronom.writeToCsv("pronomCoref_v1", false);

        System.exit(0);

        Map<Mention, AttrStruct> attrDict =
                ClassifyUtil.attributeAttachment_agent(docSet);
        List<String> ll_attr = new ArrayList<>();
        for(Mention m : attrDict.keySet()){
            AttrStruct attr = attrDict.get(m);
            Collection<Mention> attrs = attr.getAttributeMentions();
            if(!attrs.isEmpty()){
                String docID = m.getDocID();
                int capIdx = m.getCaptionIdx();
                int idx = m.getIdx();
                StringBuilder sb = new StringBuilder();
                sb.append(docID);
                sb.append("#");
                sb.append(capIdx);
                sb.append(",");
                sb.append(m.toString().replace(",", ""));
                sb.append(",");
                sb.append(idx);
                sb.append(",");
                for(Mention m_attr : attrs){
                    sb.append(m_attr.toString().replace(",", ""));
                    sb.append(",");
                    sb.append(m_attr.getIdx());
                    sb.append(",");
                }
                ll_attr.add(sb.toString());
            }
        }
        FileIO.writeFile(ll_attr, "attrAttach_v1", "csv", false);


        System.exit(0);
    }

    public static void exportDatasetLists()
    {
        Logger.setVerbose();
        Logger.log("Loading documents from DB");
        DBConnector conn = new DBConnector(Overlord.dbPath);
        Collection<Document> docSet = DocumentLoader.getDocumentSet(conn, 1);

        Logger.log("Building lists");
        DoubleDict<String> detDict = new DoubleDict<>();
        for(Document d : docSet)
            for(Caption c : d.getCaptionList())
                for(Token t : c.getTokenList())
                    if(t.getPosTag().equals("DT"))
                        detDict.increment(t.getText().toLowerCase());
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
        for(Document d : docSet)
            for(int i=0; i<d.getMentionList().size(); i++)
                for(int j=i+1; j<d.getMentionList().size(); j++)
                    totalPairs++;
        int pairIdx = 0;

        for(Document d : docSet){
            List<Mention> mentionList = d.getMentionList();
            for(int i=0; i<mentionList.size(); i++){
                Mention m1 = mentionList.get(i);
                Caption c1 = d.getCaption(m1.getCaptionIdx());

                String head_1 = m1.getHead().getText().toLowerCase().replace(",", "");
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
                    String head_2 = m2.getHead().getText().toLowerCase().replace(",", "");
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

    public static void pronomCorefEval(Collection<Document> docSet)
    {
        Logger.log("Storing caption dict");
        Map<String, Caption> captionDict = new HashMap<>();
        for(Document d : docSet)
            for(Caption c : d.getCaptionList())
                captionDict.put(d.getID() + "#" + c.getIdx(), c);

        Logger.log("Rule-based pronominal coref");
        Map<String, Set<Mention[]>> pronomCorefDict =
                ClassifyUtil.pronominalCoref(docSet);

        ScoreDict<String> scoreDict = new ScoreDict<>();
        Set<Mention> unattachedPronoms_pred = new HashSet<>();
        Set<Mention> unattachedPronoms_gold = new HashSet<>();
        Set<Mention[]> mismatchEx = new HashSet<>();
        for(Document d : docSet){
            //convert our pairs to a [pronom -> ante] mapping
            Set<Mention[]> pronomPairs = pronomCorefDict.get(d.getID());
            Map<Mention, Mention> predPronomAnteDict = new HashMap<>();
            if(pronomPairs == null)
                pronomPairs = new HashSet<>();
            for(Mention[] pair : pronomPairs)
                predPronomAnteDict.put(pair[1], pair[0]);

            for(Caption c : d.getCaptionList()){
                for(int i=0; i<c.getMentionList().size(); i++){
                    Mention m_pronom = c.getMentionList().get(i);
                    if(m_pronom.getPronounType() != Mention.PRONOUN_TYPE.NONE){
                        Mention pred_ante = null;
                        if(predPronomAnteDict.containsKey(m_pronom))
                            pred_ante = predPronomAnteDict.get(m_pronom);

                        //Each pronoun can take any non-pronominal antecedent
                        //or no antecedent
                        Mention gold_ante = null;
                        for(int j=i-1; j>=0; j--){
                            Mention m_ante = c.getMentionList().get(j);
                            if(m_ante.getPronounType() == Mention.PRONOUN_TYPE.NONE){
                                String pred = "0";
                                if(pred_ante != null)
                                    pred = pred_ante.equals(m_ante) ? "1" : "0";
                                String gold = "0";
                                if(m_pronom.getChainID().equals(m_ante.getChainID())){
                                    gold = "1";

                                    //prefer storing correct predictions
                                    if(gold_ante == null || pred.equals(gold))
                                        gold_ante = m_ante;
                                }
                                scoreDict.increment(gold, pred);
                            }
                        }

                        if(gold_ante == null)
                            unattachedPronoms_gold.add(m_pronom);
                        if(pred_ante == null)
                            unattachedPronoms_pred.add(m_pronom);

                        if(gold_ante != null && !gold_ante.equals(pred_ante) ||
                           pred_ante != null && !pred_ante.equals(gold_ante)){
                            mismatchEx.add(new Mention[]{pred_ante, gold_ante, m_pronom});
                        }
                    }
                }
            }
        }

        Logger.log("Saving examples in ex_pronomCoref");
        OutTable ot = new OutTable("pred_ante", "pred_ante_idx", "gold_ante", "gold_ante_idx", "pronom", "pronom_idx", "caption");
        for(Mention[] triplet : mismatchEx){
            String pred_ante = "";
            String pred_ante_idx = "";
            if(triplet[0] != null){
                pred_ante = triplet[0].toString();
                pred_ante_idx = ""+triplet[0].getIdx();
            }
            String gold_ante = "";
            String gold_ante_idx = "";
            if(triplet[1] != null){
                gold_ante = triplet[1].toString();
                gold_ante_idx = ""+triplet[1].getIdx();
            }
            Mention pronom = triplet[2];
            Caption c = captionDict.get(pronom.getDocID() + "#" + pronom.getCaptionIdx());
            ot.addRow(pred_ante, pred_ante_idx, gold_ante, gold_ante_idx,
                       pronom.toString(), pronom.getIdx(), c.toString());
        }
        ot.writeToCsv("ex_pronomCoref", true);

        System.out.println("Overall pronom coref score");
        System.out.printf("Acc: %.2f\\%% (%d of %d true in gold)\n",
                100.0 * scoreDict.getAccuracy(),
                scoreDict.getGoldCount("1"),
                scoreDict.getGoldCount("0") + scoreDict.getGoldCount("1"));
        System.out.println("1 & " + scoreDict.getScore("1").toLatexString());
        System.out.println("0 & " + scoreDict.getScore("0").toLatexString());

        Set<Mention> unattachedIntersect = new HashSet<>(unattachedPronoms_gold);
        unattachedIntersect.retainAll(unattachedPronoms_pred);
        int unatt_pred = unattachedPronoms_pred.size();
        int unatt_gold = unattachedPronoms_gold.size();
        int unatt_correct = unattachedIntersect.size();
        System.out.printf("Unattached: %d pred; %d gold; %d correct\n",
                unatt_pred, unatt_gold, unatt_correct);
        double p = 100.0 * unatt_correct / unatt_pred;
        double r = 100.0 * unatt_correct / unatt_gold;
        double a = (2 * p * r) / (p + r);
        System.out.printf("%.2f\\%% & %.2f\\%% & %.2f\\%%\n",
                p, r, a);
    }

    public static void pronomSubsetEval(Collection<Document> docSet)
    {
        Logger.log("Storing caption dict");
        Map<String, Caption> captionDict = new HashMap<>();
        for(Document d : docSet)
            for(Caption c : d.getCaptionList())
                captionDict.put(d.getID() + "#" + c.getIdx(), c);

        Logger.log("Rule-based intra-caption subset pred");
        Map<String, Set<Mention[]>> subsetDict =
                ClassifyUtil.pronomSubset(docSet);

        Logger.log("Excluding pronom coref");
        Map<String, Set<Mention[]>> pronomCoref =
                ClassifyUtil.pronominalCoref(docSet);
        Map<String, Set<Mention[]>> subsetDict_sansPronom =
                new HashMap<>();
        for(String docID : subsetDict.keySet()){
            Set<Mention[]> subSupPairs = new HashSet<>();

            //we want to throw out all pairs that pronom coref
            //has fired on
            Set<Mention> attachedPronouns = new HashSet<>();
            if(pronomCoref.containsKey(docID))
                for(Mention[] pair : pronomCoref.get(docID))
                    attachedPronouns.add(pair[1]);

            //remove any predicted subset pairs with pronouns that
            //we know are coref
            for(Mention[] pair : subsetDict.get(docID))
                if(!attachedPronouns.contains(pair[1]))
                    subSupPairs.add(pair);

            if(!subSupPairs.isEmpty())
                subsetDict_sansPronom.put(docID, subSupPairs);
        }

        Logger.log("Evaluating");
        ScoreDict<Integer> scoreDict = new ScoreDict<>();
        Set<Mention[]> mismatchEx = new HashSet<>(); //pred_sup; pred_sub; gold_sup; gold_sub
        for(Document d : docSet){
            //get our pairs for this document
            Map<Mention, Mention> subSuperDict = new HashMap<>();
            if(subsetDict_sansPronom.containsKey(d.getID()))
                for(Mention[] pair : subsetDict_sansPronom.get(d.getID()))
                    subSuperDict.put(pair[1], pair[0]);

            //iterate through our caps, looking for our true labels
            //and constrasting with our predicted
            for(Caption c : d.getCaptionList()){
                for(int i=0; i<c.getMentionList().size(); i++){
                    Mention m1 = c.getMentionList().get(i);
                    for(int j=i-1; j>=0; j--){
                        Mention m2 = c.getMentionList().get(j);

                        //We only care about pairs where one is a pronoun
                        if(m1.getPronounType() == Mention.PRONOUN_TYPE.NONE && m2.getPronounType() == Mention.PRONOUN_TYPE.NONE)
                            continue;

                        //pred ji is always 0; our heuristics only work
                        //with antecendents
                        int pred_ij = 0, pred_ji = 0;
                        if(subSuperDict.containsKey(m1))
                            pred_ij = subSuperDict.get(m1).equals(m2) ? 1 : 0;
                        if(subSuperDict.containsKey(m2))
                            pred_ji = subSuperDict.get(m2).equals(m1) ? 1 : 0;
                        int gold_ij = d.getBoxesAreSubset(m1, m2) &&
                                Mention.getLexicalTypeMatch(m1,m2) > 0 ? 1 : 0;
                        int gold_ji = d.getBoxesAreSubset(m2, m1) &&
                                Mention.getLexicalTypeMatch(m1,m2) > 0 ? 1 : 0;
                        scoreDict.increment(gold_ij, pred_ij);
                        scoreDict.increment(gold_ji, pred_ji);


                        if(pred_ij != gold_ij || pred_ji != gold_ji){
                            Mention pred_sup = null;
                            Mention pred_sub = null;
                            if(pred_ij == 1) {
                                pred_sup = m2;
                                pred_sub = m1;
                            } else if(pred_ji == 1){
                                pred_sup = m1;
                                pred_sub = m2;
                            }
                            Mention gold_sup = null;
                            Mention gold_sub = null;
                            if(gold_ij == 1) {
                                gold_sup = m2;
                                gold_sub = m1;
                            } else if(gold_ji == 1) {
                                gold_sup = m1;
                                gold_sub = m2;
                            }
                            mismatchEx.add(new Mention[]{pred_sup, pred_sub, gold_sup, gold_sub});
                        }
                    }
                }
            }
        }

        Logger.log("Saving examples in ex_intraCapSubset");
        OutTable ot = new OutTable("pred_sup", "pred_sub", "gold_sup", "gold_sub", "caption");
        for(Mention[] quartet : mismatchEx){
            String[] outRow = new String[5];
            String capID = "";
            for(int i=0; i<quartet.length; i++) {
                if (quartet[i] != null) {
                    capID = quartet[i].getDocID() + "#" + quartet[i].getCaptionIdx();
                    outRow[i] = quartet[i].toString();
                }
            }
            outRow[4] = captionDict.get(capID).toString();
            ot.addRow(outRow);
        }
        ot.writeToCsv("ex_intraCapSubset", true);

        System.out.println("Overall subset score");
        System.out.printf("Acc: %.2f\\%% (%d of %d true in gold)\n",
                scoreDict.getAccuracy(), scoreDict.getGoldCount(1),
                scoreDict.getGoldCount(0) + scoreDict.getGoldCount(1));
        System.out.println("1 & " + scoreDict.getScore(1).toLatexString());
        System.out.println("0 & " + scoreDict.getScore(0).toLatexString());
    }

    public static void switchLabelsToNull(Collection<Document> docSet, String corefFeatsFile)
    {
        Logger.log("Reading coref feature vectors");
        List<String> fvStrList = FileIO.readFile_lineList(corefFeatsFile);
        Map<String, FeatureVector> fvDict = new HashMap<>();
        for(String fvStr : fvStrList){
            FeatureVector fv = FeatureVector.parseFeatureVector(fvStr);
            fvDict.put(fv.comments.trim(), fv);
        }

        Logger.log("Replacing coref labels with null labels");
        for(Document d : docSet){
            for(int i=0; i<d.getMentionList().size(); i++){
                Mention m1 = d.getMentionList().get(i);
                if(m1.getPronounType() != Mention.PRONOUN_TYPE.NONE)
                    continue;

                Set<BoundingBox> boxSet1 = d.getBoxSetForMention(m1);
                for(int j=i+1; j<d.getMentionList().size(); j++){
                    Mention m2 = d.getMentionList().get(j);
                    if(m2.getPronounType() != Mention.PRONOUN_TYPE.NONE)
                        continue;

                    Set<BoundingBox> boxSet2 = d.getBoxSetForMention(m2);

                    //find the presence of the subset relation
                    Set<BoundingBox> intersect = new HashSet<>(boxSet1);
                    intersect.retainAll(boxSet2);
                    boolean subsetRel = false;
                    if(intersect.size() == boxSet1.size() && boxSet2.size() > intersect.size())
                        subsetRel = true;
                    else if(intersect.size() == boxSet2.size() && boxSet1.size() > intersect.size())
                        subsetRel = true;

                    //get the feature vector for this pair
                    FeatureVector fv = fvDict.get(Document.getMentionPairStr(m1, m2, true));

                    //positive labels are only between links that arent subset
                    //in either direction and aren't coreferent
                    if(fv.label == 0 && !subsetRel)
                        fv.label = 1;
                    else
                        fv.label = 0;
                }
            }
        }

        Logger.log("Writing the new features");
        String nullFile = corefFeatsFile.replace("coref", "null");
        nullFile = nullFile.replace(".feats", "");
        List<String> nullLineList = new ArrayList<>();
        fvDict.values().forEach(fv -> nullLineList.add(fv.toString()));
        FileIO.writeFile(nullLineList, nullFile, "feats", false);
    }

    public static void boxSubsetEx()
    {
        //get train and dev from the db
        DBConnector conn = new DBConnector(Overlord.dbPath);
        Collection<Document> docSet = DocumentLoader.getDocumentSet(conn, 1);
        docSet.addAll(DocumentLoader.getDocumentSet(conn, 0));

        OutTable ot = new OutTable("img_id", "m_sub", "m_sub_debug", "m_super", "m_super_debug", "case", "caption");
        for(Document d : docSet){
            for(Caption c : d.getCaptionList()){
                Map<Mention[], String> subsetPairCaseDict = new HashMap<>();
                for(int i=1; i<c.getMentionList().size(); i++){
                    Mention m1 = c.getMentionList().get(i);
                    Set<BoundingBox> boxes_1 = d.getBoxSetForMention(m1);
                    for(int j=i-1; j >= 0; j--){
                        Mention m2 = c.getMentionList().get(j);
                        Set<BoundingBox> boxes_2 = d.getBoxSetForMention(m2);

                        //Subset mentions must be _either_ of matching types _or_ are pronouns
                        if(m1.getPronounType() != Mention.PRONOUN_TYPE.NONE ||
                           m1.getLexicalType().equals(m2.getLexicalType())){
                            Set<BoundingBox> intersect = new HashSet<>(boxes_1);
                            intersect.retainAll(boxes_2);

                            Mention[] pair = {m1,m2};
                            String caseStr = null;
                            //we only care if they both have boxes
                            if(!boxes_1.isEmpty() && !boxes_2.isEmpty()){
                                if(boxes_2.containsAll(boxes_1) && boxes_2.size() > intersect.size()) {
                                    caseStr = "proper_subset";
                                } else {
                                    //if this doesn't have a proper subset
                                    //relationship, check a few other cases,
                                    //including IOU and equality
                                    if(boxes_2.containsAll(boxes_1) && boxes_2.size() == boxes_1.size()) {
                                        if(boxes_1.size() == 1){
                                            caseStr = "identity_single";
                                        } else {
                                            caseStr = "identity_multi";
                                        }
                                    } else {
                                        //We want to account for boxes b1 that overlap
                                        //with high IOU with boxes b2, but we want to
                                        //ignore cases where many b1s are stacked on
                                        //a single b2
                                        double iouThresh = 1.0;
                                        while(caseStr == null && iouThresh >= .8){
                                            Map<BoundingBox, Set<BoundingBox>> b1ToB2Dict = new HashMap<>();
                                            for(BoundingBox b1 : boxes_1){
                                                for(BoundingBox b2 : boxes_2){
                                                    if(BoundingBox.IOU(b1, b2) >= iouThresh){
                                                        if(!b1ToB2Dict.containsKey(b1))
                                                            b1ToB2Dict.put(b1, new HashSet<>());
                                                        b1ToB2Dict.get(b1).add(b2);
                                                    }
                                                }
                                            }

                                            //before worrying about the stacking case, determine
                                            //if there's even a subset relationship here;
                                            //If all boxes in b1 have a b2 mapping and b1 is smaller
                                            //than b2, this could be a proper subset
                                            if(boxes_1.size() == b1ToB2Dict.keySet().size() &&
                                               boxes_1.size() < boxes_2.size())
                                            {
                                                //to ignore the stacking case, make a pairwise
                                                //comparison between each set of associated
                                                //bounding boxes; if there aren't at least
                                                //one disjoint element per set, this is a stacking
                                                //case
                                                boolean foundStack = false;
                                                List<Set<BoundingBox>> assocBoxes = new ArrayList<>(b1ToB2Dict.values());
                                                for(int l=0; l<assocBoxes.size(); l++){
                                                    for(int m=l+1; m<assocBoxes.size(); m++){
                                                        Set<BoundingBox> assoc_1 = assocBoxes.get(l);
                                                        Set<BoundingBox> assoc_2 = assocBoxes.get(m);
                                                        Set<BoundingBox> assocIntersect = new HashSet<>(assoc_1);
                                                        assocIntersect.retainAll(assoc_2);
                                                        if(assoc_1.size() == assocIntersect.size() ||
                                                           assoc_2.size() == assocIntersect.size()){
                                                            foundStack = true;
                                                            break;
                                                        }
                                                    }
                                                    if(foundStack)
                                                        break;
                                                }
                                                if(!foundStack){
                                                    caseStr = "IOU_" + iouThresh;
                                                }
                                            }

                                            iouThresh -= 0.05;
                                        }
                                    }
                                }
                            }

                            if(caseStr != null){
                                subsetPairCaseDict.put(pair, caseStr);
                            }
                        }
                    }
                }

                for(Mention[] pair : subsetPairCaseDict.keySet()){
                    ot.addRow(d.getID(), pair[0].toString(), pair[0].toDebugString(), pair[1].toString(), pair[1].toDebugString(),
                              subsetPairCaseDict.get(pair), c.toString());
                }
            }
        }
        ot.writeToCsv("ex_subsets");
    }

    public static void nonvisEx()
    {
        Set<String> nonvisLex =
                new HashSet<>(Arrays.asList(new String[]{"background", "air",
                "picture", "camera", "trick", "ground",
                "day", "side", "photo", "distance",
                "jump", "midair", "music", "night",
                "view", "foreground"}));

        DBConnector conn = new DBConnector(Overlord.dbPath);
        Collection<Document> docSet = DocumentLoader.getDocumentSet(conn, 0);

        OutTable ot = new OutTable("cap_id", "pred", "gold", "correct", "pred_case", "caption", "caption_coref");
        BinaryClassifierScoreDict bcScoreDict =
                new BinaryClassifierScoreDict("../flickr30kEntities_v2_nonvis_dev.scores");
        Map<String, ScoreDict<Integer>> caseScoreDict = new HashMap<>();
        caseScoreDict.put("combined", new ScoreDict<>());
        caseScoreDict.put("model", new ScoreDict<>());
        caseScoreDict.put("heuristic", new ScoreDict<>());

        for(Document d : docSet){
            for(Caption c : d.getCaptionList()){
                for(Mention m : c.getMentionList()){
                    int gold = 0;
                    if(m.getChainID().equals("0"))
                        gold = 1;

                    int pred_model = 0;
                    if(bcScoreDict.get(m) != null && bcScoreDict.get(m) > 0)
                        pred_model = 1;

                    int pred_heur = 0;
                    String normLemma = m.getHead().getLemma().toLowerCase().trim();
                    if(nonvisLex.contains(normLemma))
                        pred_heur = 1;

                    caseScoreDict.get("model").increment(gold, pred_model);
                    caseScoreDict.get("heuristic").increment(gold, pred_heur);
                    caseScoreDict.get("combined").increment(gold, pred_model + pred_heur > 0 ? 1 : 0);

                    if(pred_heur + pred_model > 0){
                        String goldStr = "";
                        String predStr = "";
                        if(pred_heur + pred_model > 0)
                            predStr = m.toString();
                        if(gold == 1)
                            goldStr = m.toString();
                        String caseStr = "";
                        if(pred_model == 1 && pred_heur == 1)
                            caseStr = "both";
                        else if(pred_model == 1)
                            caseStr = "model";
                        else if(pred_heur == 1)
                            caseStr = "heuristic";
                        int correct = gold == Math.min(1, pred_heur + pred_model) ? 1 : 0;
                        if(!goldStr.isEmpty() || !predStr.isEmpty())
                            ot.addRow(c.getUniqueID(), predStr, goldStr, correct, caseStr, c.toString(), c.toCorefString());
                    }
                }
            }
        }

        for(String caseStr : caseScoreDict.keySet()){
            System.out.println("---" + caseStr + "---");
            ScoreDict<Integer> scoreDict = caseScoreDict.get(caseStr);
            System.out.println("0: " + scoreDict.getScore(0).toScoreString());
            System.out.println("1: " + scoreDict.getScore(1).toScoreString());
            System.out.printf("acc: %.3f%% (of %d)\n", 100.0 * scoreDict.getAccuracy(),
                    scoreDict.getTotalGold());
            System.out.println();

            System.out.println("0 & " + scoreDict.getScore(0).toLatexString());
            System.out.println("1 & " + scoreDict.getScore(1).toLatexString());
        }
        ot.writeToCsv("ex_nonvis");
    }

    public static void subsetEval(Collection<Document> docSet, String featsFile, String scoresFile)
    {
        //Store our scores as a map, since we can't use the binary score dict, given the enforced order
        Logger.log("Reading .scores file");
        Map<String, Double> subsetScoreDict = new HashMap<>();
        String[][] scoreTable = FileIO.readFile_table(scoresFile);
        for(String[] row : scoreTable)
            subsetScoreDict.put(row[0], Double.parseDouble(row[1]));

        Logger.log("Getting true labels from .feats file");
        Map<String, Integer> subsetLabelDict = new HashMap<>();
        List<String> featStrList = FileIO.readFile_lineList(featsFile);
        for(String featStr : featStrList){
            FeatureVector fv = FeatureVector.parseFeatureVector(featStr);
            subsetLabelDict.put(fv.comments, (int)fv.label);
        }

        Logger.log("Evaluating with heuristics");
        List<String> capList = new ArrayList<>();
        ScoreDict<Integer> scoreDict_heuristic = new ScoreDict<>();
        ScoreDict<Integer> scoreDict_model = new ScoreDict<>();
        ScoreDict<Integer> scoreDict = new ScoreDict<>();
        for(Document d : docSet){
            for(Caption c : d.getCaptionList()){
                for(int i=0; i<c.getMentionList().size(); i++){
                    Mention m1 = c.getMentionList().get(i);
                    for(int j=i+1; j<c.getMentionList().size(); j++){
                        Mention m2 = c.getMentionList().get(j);

                        boolean isPronom1 = m1.getPronounType() != Mention.PRONOUN_TYPE.NONE;
                        boolean isPronom2 = m2.getPronounType() != Mention.PRONOUN_TYPE.NONE;
                        boolean isAgent1 = m1.getLexicalType().contains("people") || m1.getLexicalType().contains("animals");
                        boolean isAgent2 = m2.getLexicalType().contains("people") || m2.getLexicalType().contains("animals");
                        Set<String> types1 = new HashSet<>(Arrays.asList(m1.getLexicalType().split("/")));
                        Set<String> types2 = new HashSet<>(Arrays.asList(m2.getLexicalType().split("/")));
                        Set<String> typeIntersect = new HashSet<>(types1);
                        typeIntersect.retainAll(types2);
                        boolean sameType = !typeIntersect.isEmpty();

                        if(isAgent1 || isAgent2){
                            String inOrderID = Document.getMentionPairStr(m1,m2,true,true);
                            String outOrderID = Document.getMentionPairStr(m2,m1,true,true);
                            int inOrderGold = subsetLabelDict.get(inOrderID);
                            int outOrderGold = subsetLabelDict.get(outOrderID);
                            int inOrderPred_model = subsetScoreDict.get(inOrderID) > 0 ? 1 : 0;
                            int outOrderPred_model = subsetScoreDict.get(outOrderID) > 0 ? 1 : 0;
                            int inOrderPred_heuristics = 0;
                            int outOrderPred_heuristics = 0;
                            if(sameType || isPronom1 || isPronom2){
                                if(!isPronom1 && isAgent1 && (m2.toString().equalsIgnoreCase("they") || m2.toString().equalsIgnoreCase("them"))){
                                    inOrderPred_heuristics = 1;
                                } //attach x of y if they're the same type or one is a pronoun
                                else if((sameType || isPronom1 || isPronom2) &&
                                        (m1.getTokenRange()[1] + 2 == m2.getTokenRange()[0]) &&
                                        (c.getTokenList().get(m1.getTokenRange()[1]+1).getText().equalsIgnoreCase("of"))){
                                    inOrderPred_heuristics = 1;
                                } //attach all cases of X... one
                                else if(isAgent1 && m1.getCardinality().getValue() > 1 && m2.toString().equalsIgnoreCase("one")){
                                    outOrderPred_heuristics = 1;
                                }
                            }

                            scoreDict_model.increment(inOrderGold, inOrderPred_model);
                            scoreDict_model.increment(outOrderGold, outOrderPred_model);
                            scoreDict_heuristic.increment(inOrderGold, inOrderPred_heuristics);
                            scoreDict_heuristic.increment(outOrderGold, outOrderPred_heuristics);

                            int inOrderPred = inOrderPred_heuristics > 0 || inOrderPred_model > 0 ? 1 : 0;
                            int outOrderPred = outOrderPred_heuristics > 0 || outOrderPred_model > 0 ? 1 : 0;
                            scoreDict.increment(inOrderGold, inOrderPred);
                            scoreDict.increment(outOrderGold, outOrderPred);

                            if(inOrderGold != inOrderPred){
                                String capStr = String.format("%d | %s (%s) sub %s (%s) | %s",
                                        inOrderGold, m1.toString(), m1.getLexicalType(),
                                        m2.toString(), m2.getLexicalType(), c.toString());
                                capList.add(capStr);
                            } else if(outOrderGold != outOrderPred){
                                String capStr = String.format("%d | %s (%s) sub %s (%s) | %s",
                                        outOrderGold, m2.toString(), m2.getLexicalType(),
                                        m1.toString(), m1.getLexicalType(), c.toString());
                                capList.add(capStr);
                            }
                        }
                    }
                }
            }
        }

        FileIO.writeFile(capList, "ex_subset_caps_incorrect");

        Logger.log("Scores (model)");
        System.out.println("1: " + scoreDict_model.getScore(1).toLatexString());
        System.out.println("0: " + scoreDict_model.getScore(0).toLatexString());
        System.out.printf("Acc: %.2f%% (of %d; %d positive in gold)\n",
                100.0 * scoreDict_model.getAccuracy(), scoreDict_model.getTotalGold(),
                scoreDict_model.getGoldCount(1));

        Logger.log("Scores (heuristics)");
        System.out.println("1: " + scoreDict_heuristic.getScore(1).toLatexString());
        System.out.println("0: " + scoreDict_heuristic.getScore(0).toLatexString());
        System.out.printf("Acc: %.2f%% (of %d; %d positive in gold)\n",
                100.0 * scoreDict_heuristic.getAccuracy(), scoreDict_heuristic.getTotalGold(),
                scoreDict_heuristic.getGoldCount(1));

        Logger.log("Scores");
        System.out.println("1: " + scoreDict.getScore(1).toLatexString());
        System.out.println("0: " + scoreDict.getScore(0).toLatexString());
        System.out.printf("Acc: %.2f%% (of %d; %d positive in gold)\n",
                100.0 * scoreDict.getAccuracy(), scoreDict.getTotalGold(),
                scoreDict.getGoldCount(1));
    }

    public static void corefEval(Collection<Document> docSet, String scoresFile, String plusAttrScoreFile)
    {
        Map<String, Set<Mention[]>> pronomCorefDict = ClassifyUtil.pronominalCoref(docSet);
        BinaryClassifierScoreDict pcDict = new BinaryClassifierScoreDict(scoresFile);
        BinaryClassifierScoreDict pcDict_plusAttr = new BinaryClassifierScoreDict(plusAttrScoreFile);
        ScoreDict<Integer> corefScores = new ScoreDict<>();
        ScoreDict<Integer> plusPronomScores = new ScoreDict<>();
        ScoreDict<Integer> plusAttrScores = new ScoreDict<>();
        ScoreDict<Integer> plusPronomAttrScores = new ScoreDict<>();
        for(Document d : docSet){
            //base coref
            Map<Mention, Integer> mentionChainDict =
                    getMentionChainDict(d, pcDict, null);
            Set<Integer[]> goldPredSet = getGoldPredSet(mentionChainDict);
            for(Integer[] goldPred : goldPredSet)
                corefScores.increment(goldPred[0], goldPred[1]);

            // + pronom coref
            Set<Mention[]> pronomCoref = pronomCorefDict.get(d.getID());
            mentionChainDict = getMentionChainDict(d, pcDict, pronomCoref);
            goldPredSet = getGoldPredSet(mentionChainDict);
            for(Integer[] goldPred : goldPredSet)
                plusPronomScores.increment(goldPred[0], goldPred[1]);

            // + attr
            mentionChainDict = getMentionChainDict(d, pcDict_plusAttr, null);
            goldPredSet = getGoldPredSet(mentionChainDict);
            for(Integer[] goldPred : goldPredSet)
                plusAttrScores.increment(goldPred[0], goldPred[1]);

            // + pronom + attr
            mentionChainDict = getMentionChainDict(d, pcDict_plusAttr, pronomCoref);
            goldPredSet = getGoldPredSet(mentionChainDict);
            for(Integer[] goldPred : goldPredSet)
                plusPronomAttrScores.increment(goldPred[0], goldPred[1]);
        }

        Logger.log("Scores");
        System.out.println("1: " + corefScores.getScore(1).toLatexString());
        System.out.println("0: " + corefScores.getScore(0).toLatexString());
        System.out.printf("Acc: %.2f%% (of %d; %d positive in gold)\n",
                100.0 * corefScores.getAccuracy(), corefScores.getTotalGold(),
                corefScores.getGoldCount(1));

        Logger.log("Scores (+ pronom)");
        System.out.println("1: " + plusPronomScores.getScore(1).toLatexString());
        System.out.println("0: " + plusPronomScores.getScore(0).toLatexString());
        System.out.printf("Acc: %.2f%% (of %d; %d positive in gold)\n",
                100.0 * plusPronomScores.getAccuracy(), plusPronomScores.getTotalGold(),
                plusPronomScores.getGoldCount(1));


        Logger.log("Scores (+ attr)");
        System.out.println("1: " + plusAttrScores.getScore(1).toLatexString());
        System.out.println("0: " + plusAttrScores.getScore(0).toLatexString());
        System.out.printf("Acc: %.2f%% (of %d; %d positive in gold)\n",
                100.0 * plusAttrScores.getAccuracy(), plusAttrScores.getTotalGold(),
                plusAttrScores.getGoldCount(1));


        Logger.log("Scores (+ pronom + attr)");
        System.out.println("1: " + plusPronomAttrScores.getScore(1).toLatexString());
        System.out.println("0: " + plusPronomAttrScores.getScore(0).toLatexString());
        System.out.printf("Acc: %.2f%% (of %d; %d positive in gold)\n",
                100.0 * plusPronomAttrScores.getAccuracy(), plusPronomAttrScores.getTotalGold(),
                plusPronomAttrScores.getGoldCount(1));
    }

    private static Set<Integer[]> getGoldPredSet(Map<Mention, Integer> mentionChainDict)
    {
        Set<Integer[]> goldPredSet = new HashSet<>();
        List<Mention> mSet = new ArrayList<>(mentionChainDict.keySet());
        for(int i=0; i<mSet.size(); i++){
            Mention m1 = mSet.get(i);
            if(m1.getChainID().equals("0"))
                continue;
            for(int j=i+1; j<mSet.size(); j++){
                Mention m2 = mSet.get(j);
                if(m2.getChainID().equals("0"))
                    continue;;
                int gold = m1.getChainID().equals(m2.getChainID()) ? 1 : 0;
                int pred = mentionChainDict.get(m1).equals(mentionChainDict.get(m2)) ? 1 : 0;
                goldPredSet.add(new Integer[]{gold, pred});
            }
        }
        return goldPredSet;
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

        for(int i=0; i<d.getMentionList().size(); i++){
            Mention m1 = d.getMentionList().get(i);
            if(m1.getChainID().equals("0"))
                continue;
            for(int j=i+1; j<d.getMentionList().size(); j++){
                Mention m2 = d.getMentionList().get(j);
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

    public static void augmentCorefFeatsWithAttr(Collection<Document> docSet, String trainFeats, String devFeats)
    {
        Logger.log("Attaching attributes");
        Map<Mention, AttrStruct> mentionAttrDict =
                ClassifyUtil.attributeAttachment_agent(docSet);

        Logger.log("Writing example file for %d train mentions with attributes", mentionAttrDict.size());
        List<String> lineList = new ArrayList<>();
        for(Mention m : mentionAttrDict.keySet()){
            lineList.add(m.getUniqueID());
            lineList.add(mentionAttrDict.get(m).toLatexString());
        }
        FileIO.writeFile(lineList, "ex_attr");

        Logger.log("Reading train coref feat vectors");
        Map<String, FeatureVector> fvDict = new HashMap<>();
        List<String> trainVectorStrList = FileIO.readFile_lineList(trainFeats);
        for(String trainVecStr : trainVectorStrList){
            FeatureVector fv = FeatureVector.parseFeatureVector(trainVecStr);
            fvDict.put(fv.comments, fv);
        }
        int maxFeatureIdx = -1;
        for(FeatureVector fv : fvDict.values()){
            List<Integer> indices = fv.getFeatureIndices();
            Collections.sort(indices);
            Collections.reverse(indices);
            if(indices.get(0) > maxFeatureIdx)
                maxFeatureIdx = indices.get(0);
        }
        maxFeatureIdx++;

        //get dev feats, if we've been given a dev file
        if(devFeats != null){
            Logger.log("Reading dev coref feat vectors");
            fvDict = new HashMap<>();
            List<String> devFeatVectorStrList = FileIO.readFile_lineList(devFeats);
            for(String s : devFeatVectorStrList){
                FeatureVector fv = FeatureVector.parseFeatureVector(s);
                fvDict.put(fv.comments, fv);
            }
        }

        Logger.log("Augmenting with attr_feat");
        int numAugmentedVectors = 0;
        for(Document d : docSet){
            for(int i=0; i<d.getMentionList().size(); i++){
                Mention m1 = d.getMentionList().get(i);
                for(int j=i+1; j<d.getMentionList().size(); j++){
                    Mention m2 = d.getMentionList().get(j);

                    if(mentionAttrDict.containsKey(m1) &&
                            mentionAttrDict.containsKey(m2)){
                        AttrStruct rootAttr1 = mentionAttrDict.get(m1);
                        AttrStruct rootAttr2 = mentionAttrDict.get(m2);

                        List<Double> overlapList = new ArrayList<>();
                        overlapList.add(getAttrOverlap(rootAttr1, rootAttr2, "head"));
                        overlapList.add(getAttrOverlap(rootAttr1, rootAttr2, "torso"));
                        overlapList.add(getAttrOverlap(rootAttr1, rootAttr2, "arms"));
                        overlapList.add(getAttrOverlap(rootAttr1, rootAttr2, "hands"));
                        overlapList.add(getAttrOverlap(rootAttr1, rootAttr2, "legs"));
                        overlapList.add(getAttrOverlap(rootAttr1, rootAttr2, "feet"));
                        int overlapCount = rootAttr1.getNumAttributes(rootAttr2);
                        int count1 = rootAttr1.getNumAttributes();
                        int count2 = rootAttr2.getNumAttributes();
                        overlapList.add((double)overlapCount / (double)(count1 + count2 - overlapCount));

                        String pairID = Document.getMentionPairStr(m1,m2,true);
                        if(fvDict.containsKey(pairID)){
                            numAugmentedVectors++;

                            for(Double val : overlapList){
                                if(val != 0)
                                    fvDict.get(pairID).addFeature(maxFeatureIdx, val);
                                maxFeatureIdx++;
                            }
                        }
                    }
                }
            }
        }

        Logger.log("Saving vectors (%d newly augmented)", numAugmentedVectors);
        List<String> lineList_fv = new ArrayList<>();
        for(FeatureVector fv : fvDict.values())
            lineList_fv.add(fv.toString());
        String filename = "../flickr30kEntities_v2_coref_train_attr";
        if(devFeats != null)
            filename = "../flickr30kEntities_v2_coref_dev_attr";
        FileIO.writeFile(lineList_fv, filename, "feats", false);
        Logger.log("Done");
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

    /**Exports a feature file that has - as its features - the projection of
     * boxFeat * mentionFeat / 2
     *
     * @param boxFile
     * @param mentionFile
     * @param docSet
     */
    public static void trainBoxAffinity(String boxFile, String mentionFile,
                                        Collection<Document> docSet)
    {
        Logger.log("Storing mention / box pairs; Positive labels");
        Set<String> mbPairSet_neg = new HashSet<>();
        Set<String> mbPairSet_pos = new HashSet<>();
        for(Document d : docSet){
            for(BoundingBox b : d.getBoundingBoxSet()){
                Set<Mention> mSet = d.getMentionSetForBox(b);
                for(Mention m : d.getMentionList()){
                    String pairStr = m.getUniqueID() + "|" + b.getUniqueID();
                    if(mSet.contains(m))
                        mbPairSet_pos.add(pairStr);
                    else
                        mbPairSet_neg.add(pairStr);
                }
            }
        }
        Logger.log("Positive ex: %d; Neg ex: %d", mbPairSet_pos.size(), mbPairSet_neg.size());
        Logger.log("Dropping docset");
        docSet.clear();
        docSet = null;

        Logger.log("Loading feature files into memory");
        List<String> ll_boxFeats = FileIO.readFile_lineList(boxFile);
        Map<String, FeatureVector> boxFeats = new HashMap<>();
        for(String featLine : ll_boxFeats){
            FeatureVector fv = FeatureVector.parseFeatureVector(featLine);
            boxFeats.put(fv.comments, fv);
        }
        ll_boxFeats.clear();
        List<String> ll_mentionFeats = FileIO.readFile_lineList(mentionFile);
        Map<String, FeatureVector> mentionFeats = new HashMap<>();
        for(String featLine : ll_mentionFeats){
            FeatureVector fv = FeatureVector.parseFeatureVector(featLine);
            mentionFeats.put(fv.comments, fv);
        }
        ll_mentionFeats.clear();

        Logger.log("Computing total number of vectors / max size of vectors");
        int posExCount = 0;
        int totalExCount = 0;
        for(Document d : docSet){
            for(BoundingBox b : d.getBoundingBoxSet()){
                Set<Mention> mentionSet_box = d.getMentionSetForBox(b);
                posExCount += mentionSet_box.size();
                totalExCount += d.getMentionList().size();
            }
        }
        int maxMentionFeatIdx = 0;
        int maxBoxFeatIdx = 0;
        for(FeatureVector fv : mentionFeats.values())
            for(Integer i : fv.getFeatureIndices())
                if(i > maxMentionFeatIdx)
                    maxMentionFeatIdx = i;
        for(FeatureVector fv : boxFeats.values())
            for(Integer i : fv.getFeatureIndices())
                if(i > maxBoxFeatIdx)
                    maxBoxFeatIdx = i;
        int maxFeatIdx = maxMentionFeatIdx * maxBoxFeatIdx;
        Logger.log("Found %d positive; %d total; maxIdx: %d",
                    posExCount, totalExCount, maxFeatIdx);


        Model _model = null;
        Parameter _params = new Parameter(SolverType.L2R_LR_DUAL, 1.0, 0.001);
        Problem problem = new Problem();

        List<Feature[]> xList = new ArrayList<>();
        List<Double> yList = new ArrayList<>();
        int vectorCount = 0;
        for(Document d : docSet){
            for(BoundingBox b : d.getBoundingBoxSet()){
                FeatureVector fv_box = boxFeats.get(b.getUniqueID());
                Set<Mention> boxMentions = d.getMentionSetForBox(b);

                for(Mention m : d.getMentionList()){
                    FeatureVector fv_mention = mentionFeats.get(m.getUniqueID());

                    int label = 0;
                    if(boxMentions.contains(m))
                        label = 1;

                    //project the feature vectors into a higher dimensional space
                    yList.add((double)label);
                    List<Feature> featureList = new ArrayList<>();
                    for(int bIdx : fv_box.getFeatureIndices()){
                        for(int mIdx : fv_mention.getFeatureIndices()){
                            int projIdx = (bIdx - 1) * maxMentionFeatIdx;
                            projIdx += mIdx;
                            featureList.add(new FeatureNode(projIdx,
                                    fv_box.getFeatureValue(bIdx) *
                                            fv_mention.getFeatureValue(mIdx) / 2));
                        }
                    }
                    Feature[] fArr = new Feature[featureList.size()];
                    xList.add(featureList.toArray(fArr));
                    vectorCount++;
                    Logger.logStatus("Stored %d vectors (%.2f%%)",
                            vectorCount, 100.0 * (double)vectorCount / totalExCount);
                }
            }
        }

        //down convert to primitives
        Feature[][] x = new Feature[xList.size()][];
        x = xList.toArray(x);
        double[] y = new double[yList.size()];
        for(int i=0; i<yList.size(); i++)
            y[i] = yList.get(i);

        //set the problem vars
        problem.l = yList.size();
        problem.n = maxFeatIdx;
        problem.x = x;
        problem.y = y;

        //Finally, train_excludeIndices the _model
        Logger.log("Training model");
        _model = Linear.train(problem, _params);
        Logger.log("Saving model");
        File f = new File("models/affinity.model");
        try{
            _model.save(f);
        } catch(IOException ioEx) {
            Logger.log(ioEx);
        }
        Logger.log("Done");
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
                DocumentLoader.getDocumentSet(new DBConnector(Overlord.dbPath));
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
}
