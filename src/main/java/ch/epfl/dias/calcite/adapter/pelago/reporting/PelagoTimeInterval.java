package ch.epfl.dias.calcite.adapter.pelago.reporting;


/** Class used for measuring time between two intervals */
public class PelagoTimeInterval {
    private long startTime;
    private long endTime;
    private boolean started;

    public PelagoTimeInterval(){
        started = false;
    }

    public void start() {
        started = true;
        startTime = System.nanoTime();
    }

    public void stop() {
        started = false;
        endTime = System.nanoTime();
    }

    public long getDifferenceNano() {
        return endTime-startTime;
    }

    public long getDifferenceMilli() {
        return (getDifferenceNano())/1000000;
    }

    public boolean getStarted() {
        return started;
    }
}
