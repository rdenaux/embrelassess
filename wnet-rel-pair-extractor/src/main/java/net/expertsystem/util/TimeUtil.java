package net.expertsystem.util;

public class TimeUtil {

	public static String humanTimeFrom(long start) {
		return humanTimeFromMs(System.currentTimeMillis() - start);
	}
	
	public static String humanTimeFromMs(final long ms) {
		final long hourInMs = 1000 * 60 * 60;
		final long minuteInMs = 1000 * 60;
		final long secInMs = 1000; 
		final long h = ms / hourInMs;
		final long m = (ms % hourInMs) / minuteInMs;
		final long s = (ms % minuteInMs) / secInMs;
		final long msr = (ms % secInMs);
		return String.format("%d:%02d:%02d.%d", h, m, s, msr);
	}
	
}
