package statistical;

import utilities.DoubleDict;
import utilities.Logger;

import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

/** Computes Krippendorff's Alpha for coreference chains, according to
 *  Passonneau 2004
 *
 *  Given
 *      m : the number of coders
 *      r : the number of units
 *      i : an index for units i = {1, ..., r}
 *  Assume an r x m table, as
 *           coder_0 | coder_1 | coder_m
 *          -----------------------------
 *  unit_0  |    1    |     1   |    1
 *  unit_1  |    2    |     2   |    2
 *  unit_2  |    4    |     5   |    3
 *  unit_3  |    1    |     1   |    1
 *  unit_4  |    2    |     2   |    2
 *  unit_5  |    2    |     2   |    2
 *  unit_6  |    6    |     7   |    8
 *  unit_7  |    4    |     9   |    3
 *  unit_8  |    6    |     10  |    3
 *  unit_9  |    6    |     7   |    3
 *  unit_r  |    2    |     2   |    2
 *
 *      where each cell refers to a cluster of mentions (1 = {unit_0, unit_3})
 *
 *  For every pair of table values, b and c
 *      delta(b,c) is the distance between the clusters
 *          ~ most simply, delta(b,c) = 0 if b=c; delta(b,c)=1 otherwise
 *  n_b : the number of times value b occurs in row i
 *
 *  Krippendorff's alpha, for this setting, would be given by
 *
 *  a = 1 - (norm) * d_o / d_e
 *      NOTE: this formulation isn't an accurate description of D_o or D_e, but
 *            breaking out this normalizer makes the notation easier
 *  where
 *      norm = (rm-1) / (m-1)
 *      d_o = \sum{i} \sum{b}\sum{c>b} n_bi n_ci delta(b,c)
 *          for each row
 *              for each cell value, b
 *                  for each next cell value, c
 *      d_e = \sum{b} \sum{c} n_b n_c delta(b,c)
 *          for each pair of values, b,c
 *
 * For sets, delta(b,c) should reflect the presence or absence of mentions
 * in the cluster, so let's assign distances as
 *      0.0  : identity
 *      0.33 : subsumtion
 *      0.66 : intersection
 *      1.0  : disjoint
 *
 */
@Deprecated
public class KrippendorffsAlpha
{
    public static void testKrippendorffsAlpha()
    {
        Set<Integer> oneSet = new HashSet<>();
        oneSet.add(0);
        oneSet.add(3);
        Set<Integer> twoSet = new HashSet<>();
        twoSet.add(1);
        twoSet.add(4);
        twoSet.add(5);
        twoSet.add(10);
        Set<Integer> threeSet = new HashSet<>();
        threeSet.add(2);
        threeSet.add(7);
        threeSet.add(8);
        threeSet.add(9);
        Set<Integer> fourSet = new HashSet<>();
        fourSet.add(2);
        fourSet.add(7);
        Set<Integer> fiveSet = new HashSet<>();
        fiveSet.add(2);
        Set<Integer> sixSet = new HashSet<>();
        sixSet.add(6);
        sixSet.add(8);
        sixSet.add(9);
        Set<Integer> sevenSet = new HashSet<>();
        sevenSet.add(6);
        sevenSet.add(9);
        Set<Integer> eightSet = new HashSet<>();
        eightSet.add(6);
        Set<Integer> nineSet = new HashSet<>();
        nineSet.add(7);
        Set<Integer> tenSet = new HashSet<>();
        tenSet.add(8);

        Set<Integer>[][] matrix = new Set[11][3];
        matrix[0][0] = oneSet;  matrix[0][1] = oneSet;  matrix[0][2] = oneSet;
        matrix[1][0] = twoSet;  matrix[1][1] = twoSet;  matrix[1][2] = twoSet;
        matrix[2][0] = fourSet;  matrix[2][1] = fiveSet;  matrix[2][2] = threeSet;
        matrix[3][0] = oneSet;  matrix[3][1] = oneSet;  matrix[3][2] = oneSet;
        matrix[4][0] = twoSet;  matrix[4][1] = twoSet;  matrix[4][2] = twoSet;
        matrix[5][0] = twoSet;  matrix[5][1] = twoSet;  matrix[5][2] = twoSet;
        matrix[6][0] = sixSet;  matrix[6][1] = sevenSet;  matrix[6][2] = eightSet;
        matrix[7][0] = fourSet;  matrix[7][1] = nineSet;  matrix[7][2] = threeSet;
        matrix[8][0] = sixSet;  matrix[8][1] = tenSet;  matrix[8][2] = threeSet;
        matrix[9][0] = sixSet;  matrix[9][1] = sevenSet;  matrix[9][2] = threeSet;
        matrix[10][0] = twoSet; matrix[10][1] = twoSet; matrix[10][2] = twoSet;

        Logger.log("Running Krippendorff's Alpha test");
        double a = computeAlpha(matrix, false);
        System.out.println("Expecting 0.74");
        System.out.printf("Computed  %.3f\n", a);
    }

    public static <T> double computeAlpha(Set<T>[][] matrix, boolean strictDelta)
    {
        //get our basic vars
        double r = matrix.length;
        double m = matrix[0].length;

        //compute our n dict, so we don't have to recompute n values as we
        //go through everything else
        DoubleDict<Set<T>>[] nCountDictArr = new DoubleDict[(int)r];
        for(int i=0; i<r; i++){
            nCountDictArr[i] = new DoubleDict<>();

            //NOTE: we're assuming standard set equality won't work
            //      _and_ that we don't have access to some ID mapping, so
            //      when we are counting 'how often is a set used' - the n values -
            //      we're actually counting 'how often do we see an identical set',
            //      which for the purposes of evaluation is the same thing
            for(int j=0; j<m; j++)
                nCountDictArr[i].increment(matrix[i][j],0);
            for(Set<T> key : nCountDictArr[i].keySet())
                for(int j=0; j<m; j++)
                    if(getSetDelta(key, matrix[i][j]) == 0)
                        nCountDictArr[i].increment(key);
        }

        //for expected difference, we need to find how many unique
        //values are in the row
        Map<Integer, Integer> rowUniqeValDict = new HashMap<>();
        for(int i=0; i<r; i++){
            //remember, we can't trust direct equality so we have to compare the set diff
            Set<Set<T>> rowSetDict = new HashSet<>();
            for(int j=0; j<m; j++) {
                boolean foundKey = false;
                for (Set<T> key : rowSetDict)
                    if (getSetDelta(key, matrix[i][j]) == 0)
                        foundKey = true;
                if(!foundKey)
                    rowSetDict.add(matrix[i][j]);
            }
            rowUniqeValDict.put(i, rowSetDict.size());
        }

        //for expected difference, we need n values across the entire matrix
        DoubleDict<Set<T>> uniqueValDict = new DoubleDict<>();
        for(int i=0; i<r; i++)
            for(int j=0; j<m; j++)
                uniqueValDict.increment(matrix[i][j],0);
        for(Set<T> key : uniqueValDict.keySet())
            for(int i=0; i<r; i++)
                for(int j=0; j<m; j++)
                    if(getSetDelta(key, matrix[i][j]) == 0)
                        uniqueValDict.increment(key);

        //compute our observed disagreement (well, what we're calling d_o)
        double d_o = 0.0;
        for(int i=0; i<r; i++) {
            for (int j = 0; j < m; j++) {
                for (int j_prime = j + 1; j_prime < m; j_prime++) {
                    double setDelta;
                    if (strictDelta)
                        setDelta = getSetDelta_strict(matrix[i][j], matrix[i][j_prime]);
                    else
                        setDelta = getSetDelta(matrix[i][j], matrix[i][j_prime]);
                    d_o += nCountDictArr[i].get(matrix[i][j]) *
                            nCountDictArr[i].get(matrix[i][j_prime]) *
                            setDelta;
                }
            }
        }

        //compute our expected disagreement (again, not technically this)
        double d_e = 0.0;
        for(int i=0; i<r; i++){
            for(int j=0; j<m; j++){
                for(int i_prime=j; i_prime<r; i_prime++){
                    for(int j_prime=i; j_prime<m; j_prime++){
                        double setDelta;
                        if (strictDelta)
                            setDelta = getSetDelta_strict(matrix[i][j], matrix[i_prime][j_prime]);
                        else
                            setDelta = getSetDelta(matrix[i][j], matrix[i_prime][j_prime]);
                        d_e += rowUniqeValDict.get(i) *
                                rowUniqeValDict.get(i_prime) *
                                setDelta;
                    }
                }
            }
        }

        //return our alpha val
        //possibly only m, not m-1?
        return 1 - ((m*r-1) / (m-1)) * (d_o / d_e);
    }

    private static <T> double getSetDelta(Set<T> a, Set<T> b)
    {
        boolean a_contains_b = a.containsAll(b);
        boolean b_contains_a = b.containsAll(a);
        Set<T> intersection = new HashSet<>(a);
        intersection.retainAll(b);


        if(a_contains_b && b_contains_a)
            return 0;
        else if(a_contains_b || b_contains_a)
            return 1.0/3.0;
        else if(!intersection.isEmpty())
            return 2.0/3.0;
        else
            return 1.0;
    }

    private static <T> double getSetDelta_strict(Set<T> a, Set<T> b)
    {
        double setDelta = getSetDelta(a, b);
        return setDelta == 0 ? 0 : 1;
    }
}