package out;

import utilities.FileIO;

import java.util.ArrayList;
import java.util.List;

/**OutTable is basically a wrapper for data destined
 * for CSV files; it allows for the adding of rows
 * and columns in a way that doesn't require
 * the calling functions to keep a list of lists
 *
 * @author ccervantes
 */
@Deprecated
public class OutTable {

    private List<List<String>> outTable;

    /**Creating an outtable requires a list
     * of column names
     *
     * @param columnNames
     */
    public OutTable(String... columnNames){
        outTable = new ArrayList<>();
        List<String> colNames = new ArrayList<>();
        for(String colName : columnNames)
            colNames.add(colName);
        outTable.add(colNames);
    }

    /**Add a row with string values
     *
     * @param values
     */
    public void addRow(String... values){
        List<String> outRow = new ArrayList<>();
        for(String val : values)
            outRow.add(val);
        outTable.add(outRow);
    }

    /**Add a row with object values (with special
     * null handling)
     *
     * @param values
     */
    public void addRow(Object... values){
        List<String> outRow = new ArrayList<>();
        for(Object val : values){
            if(val == null)
                outRow.add("NULL");
            else
                outRow.add(val.toString());
        }
        outTable.add(outRow);
    }

    /**Writes this table to CSV
     *
     * @param filename
     */
    public void writeToCsv(String filename)
    {
        writeToCsv(filename, true);
    }

    /**Writes this table to CSV, optionally inclues the date
     *
     * @param filename
     * @param includeDate
     */
    public void writeToCsv(String filename, boolean includeDate)
    {
        FileIO.writeFile(outTable, filename, "csv", includeDate);
    }
}
