package out;

import utilities.FileIO;

import java.util.ArrayList;
import java.util.List;

public class OutTable {

    private List<List<String>> outTable;

    public OutTable(String... columnNames){
        outTable = new ArrayList<>();
        List<String> colNames = new ArrayList<>();
        for(String colName : columnNames)
        {
            colNames.add(colName);
        }
        outTable.add(colNames);
    }

    public void addRow(String... values){
        List<String> outRow = new ArrayList<>();
        for(String val : values)
        {
            outRow.add(val);
        }
        outTable.add(outRow);
    }

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

    public void writeToCsv(String filename)
    {
        writeToCsv(filename, true);
    }

    public void writeToCsv(String filename, boolean includeDate)
    {
        FileIO.writeFile(outTable, filename, "csv", includeDate);
    }
}
