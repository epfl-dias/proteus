package ch.epfl.dias.calcite.adapter.pelago.reporting;

import ch.epfl.dias.repl.Repl;
import java.sql.Timestamp;

/** Global singleton for time measurements */
public class TimeKeeper {
    public static final TimeKeeper INSTANCE = new TimeKeeper();

    private long tPlanToJson;
    private long tPlanning;
    private long tExecutor;
    private long tCodegen;
    private long tExec;

    private Timestamp lastTimestamp;

    private TimeKeeper() {}

    public void addTexec(long time_ms) {
        tExec = time_ms;
    }

    public void addTcodegen(long time_ms) {
        tCodegen = time_ms;
    }

    public void addTplan2json(long time_ms) {
        tPlanToJson = time_ms;
    }

    public void addTexecutorTime(PelagoTimeInterval interval) {
        tExecutor = interval.getDifferenceMilli();
    }

    public void addTplanning(PelagoTimeInterval interval) {
        tPlanning = interval.getDifferenceMilli();
    }

    public void addTimestamp(){
        lastTimestamp = new Timestamp(System.currentTimeMillis());
    }

    public void refreshTable() {
        TimeKeeperTable.addTimings(tExecutor + tPlanning + tPlanToJson, tPlanning, tPlanToJson, tExecutor, tCodegen, tExec, lastTimestamp);
    }


    @Override
    public String toString() {
        String format;
        if (Repl.timingscsv()) {
            format = "Timings,%d,%d,%d,%d,%d,%d";
        } else {
            format = "Total time: %dms, "
                    + "Planning time: %dms, "
                    + "Flush plan time: %dms, "
                    + "Total executor time: %dms, "
                    + "Codegen time: %dms, "
                    + "Execution time: %dms";
        }
        return String.format(format, tExecutor + tPlanning + tPlanToJson, tPlanning, tPlanToJson, tExecutor,
                                tCodegen, tExec, lastTimestamp);
    }
}
