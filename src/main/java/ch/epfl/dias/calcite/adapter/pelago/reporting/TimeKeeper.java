package ch.epfl.dias.calcite.adapter.pelago.reporting;

import java.sql.Time;
import java.sql.Timestamp;
import java.util.ArrayList;
import java.util.List;

/** Global singleton for time measurements */
public class TimeKeeper {
    private List<Long> texec;
    private List<Long> tcodegen;
    private List<Long> tcalcite;
    private List<Timestamp> timestamps;

    private static TimeKeeper instance = null;

    private long lastTexec;
    private long lastTcodegen;
    private long lastTcalcite;
    private Timestamp lastTimestamp;

    private TimeKeeper() {
        texec = new ArrayList<>();
        tcodegen = new ArrayList<>();
        tcalcite = new ArrayList<>();
        timestamps = new ArrayList<>();
    }

    public void addTexec(long texec) {
        this.lastTexec = texec;
        this.texec.add(texec);
    }

    public void addTcodegen(long tcodegen) {
        this.lastTcodegen = tcodegen;
        this.tcodegen.add(tcodegen);
    }

    public void addTcalcite(long tcalcite) {
        this.lastTcalcite = tcalcite;
        this.tcalcite.add(tcalcite);
    }

    public void addTimestamp(){
        this.lastTimestamp = new Timestamp(System.currentTimeMillis());
        this.timestamps.add(lastTimestamp);
    }

    public static TimeKeeper getInstance(){
        if(instance == null){
            instance = new TimeKeeper();
        }

        return instance;
    }

    public void refreshTable() {
        // maybe a stack would be smarter option for getting the last element
        int sizeTexec = texec.size();
        int sizeTcodegen = tcodegen.size();
        int sizeTcalcite = tcalcite.size();
        int sizeTimestamps = timestamps.size();

        assert ((sizeTcalcite == sizeTcodegen) && (sizeTcalcite == sizeTexec) && (sizeTcodegen == sizeTimestamps));

        TimeKeeperTable.addTimings(lastTexec, lastTcodegen, lastTcalcite, lastTimestamp);
    }


    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();

        sb.append("ALL EXECUTION TIMES SINCE PELAGO STARTUP!\n");

        sb.append("Timestamps: ");
        sb.append(timestamps);
        sb.append("\n");

        sb.append("Execution times: ");
        sb.append(texec);
        sb.append("\n");

        long sum_texec = 0;
        for(long el : texec){
            sum_texec += el;
        }

        sb.append("Total execution time: ");
        sb.append(sum_texec);
        sb.append("ms \n");

        sb.append("Code generation times: ");
        sb.append(tcodegen);
        sb.append("ms \n");

        long sum_tcodegen = 0;
        for(long el : tcodegen){
            sum_tcodegen += el;
        }

        sb.append("Total codegen time: ");
        sb.append(sum_tcodegen);
        sb.append("ms \n");

        sb.append("Optimization and Calcite time: ");
        sb.append(tcalcite);
        sb.append("ms \n");

        long sum_tcalcite = 0;
        for(long el : tcalcite){
            sum_tcalcite += el;
        }

        sb.append("Total calcite time: ");
        sb.append(sum_tcalcite);
        sb.append("ms \n");

        sb.append("TOTAL TIME: ");
        sb.append(sum_texec+sum_tcodegen+sum_tcalcite);
        sb.append("ms \n");

        return(sb.toString());
    }
}
