"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { AppHeader } from "@/components/AppHeader";
import { IdleHero } from "@/components/IdleHero";
import { LoadingScreen } from "@/components/LoadingScreen";
import { ResultsScreen } from "@/components/ResultsScreen";
import { AnalysisFailed } from "@/components/AnalysisFailed";
import { ScenarioModal } from "@/components/ScenarioModal";
import { MacroOverlay } from "@/components/MacroOverlay";
import { getMacroSnapshot, scoreListing, type ScoreOutcome } from "@/lib/api";
import { LOAD_STEPS } from "@/lib/demo";
import { SAMPLE_URLS, type ListingSource } from "@/lib/listingSource";
import { useTheme } from "@/lib/useTheme";
import type { MacroSnapshot, ScoreResponse } from "@/lib/types";

type Phase = "idle" | "loading" | "done" | "failed";


export default function Page() {
  const { toggle: toggleTheme, label: themeLabel } = useTheme();

  const [phase, setPhase] = useState<Phase>("idle");
  const [url, setUrl] = useState("");
  const [source, setSource] = useState<ListingSource>("bayut");
  const [analyzedUrl, setAnalyzedUrl] = useState("");
  const [loadStep, setLoadStep] = useState(0);
  const [yieldNum, setYieldNum] = useState(0);
  const [gaugeOffset, setGaugeOffset] = useState(283);
  const [scenarioOpen, setScenarioOpen] = useState(false);
  const [macroOpen, setMacroOpen] = useState(false);

  const [macro, setMacro] = useState<MacroSnapshot | null>(null);
  const [score, setScore] = useState<ScoreResponse | null>(null);
  const [failReason, setFailReason] = useState("");

  const loadTimer = useRef<ReturnType<typeof setInterval> | null>(null);
  const analysisSeq = useRef(0);
  const countTimer = useRef<ReturnType<typeof setInterval> | null>(null);

  useEffect(() => {
    let cancelled = false;
    getMacroSnapshot().then((m) => {
      if (!cancelled) setMacro(m);
    });
    return () => {
      cancelled = true;
    };
  }, []);

  const animateYield = useCallback((target: number) => {
    if (countTimer.current) clearInterval(countTimer.current);
    const dur = 1300;
    const t0 = performance.now();
    setTimeout(() => setGaugeOffset(283 - 283 * (target / 10)), 60);
    countTimer.current = setInterval(() => {
      const p = Math.min(1, (performance.now() - t0) / dur);
      const e = 1 - Math.pow(1 - p, 3);
      setYieldNum(target * e);
      if (p >= 1 && countTimer.current) clearInterval(countTimer.current);
    }, 30);
  }, []);

  const startAnalysis = useCallback(
    (analyzeUrl: string) => {
      setPhase("loading");
      setLoadStep(0);
      setAnalyzedUrl(analyzeUrl);
      setYieldNum(0);
      setGaugeOffset(283);
      setScore(null);

      const seq = ++analysisSeq.current;
      let settled = false;
      let minElapsed = false;
      let outcome: ScoreOutcome | null = null;

      const finish = () => {
        if (!settled || !minElapsed) return;
        if (analysisSeq.current !== seq) return;
        if (loadTimer.current) clearInterval(loadTimer.current);
        if (!outcome || !outcome.ok) {
          setFailReason(outcome?.reason ?? "The analysis did not complete.");
          setPhase("failed");
          return;
        }
        setScore(outcome.data);
        setPhase("done");
        animateYield(outcome.data.scoring.actual_gross_yield * 100);
      };

      scoreListing(analyzeUrl).then((res) => {
        outcome = res;
        settled = true;
        finish();
      });

      if (loadTimer.current) clearInterval(loadTimer.current);
      let i = 0;
      loadTimer.current = setInterval(() => {
        i++;
        if (i >= LOAD_STEPS.length) {
          minElapsed = true;
          setLoadStep(LOAD_STEPS.length - 1);
          finish();
        } else {
          setLoadStep(i);
        }
      }, 750);
    },
    [animateYield]
  );

  useEffect(() => {
    return () => {
      if (loadTimer.current) clearInterval(loadTimer.current);
      if (countTimer.current) clearInterval(countTimer.current);
    };
  }, []);

  const doAnalyze = useCallback(() => {
    startAnalysis(url.trim());
  }, [url, startAnalysis]);

  const onReset = useCallback(() => {
    setPhase("idle");
    setUrl("");
  }, []);

  return (
    <div
      style={{
        minHeight: "100vh",
        background: "var(--bg)",
        transition: "background-color .35s ease",
        display: "flex",
        flexDirection: "column",
      }}
    >
      <AppHeader
        macro={macro}
        themeLabel={themeLabel}
        onToggleTheme={toggleTheme}
        onReset={onReset}
        onOpenMacro={() => setMacroOpen(true)}
      />

      {phase === "idle" && (
        <IdleHero
          url={url}
          onUrl={setUrl}
          onAnalyze={doAnalyze}
          onSample={() => startAnalysis(SAMPLE_URLS[source])}
          source={source}
          onSource={setSource}
        />
      )}

      {phase === "loading" && <LoadingScreen loadStep={loadStep} />}

      {phase === "failed" && (
        <AnalysisFailed
          url={analyzedUrl}
          reason={failReason}
          onReset={onReset}
        />
      )}

      {phase === "done" && (
        <ResultsScreen
          analyzedUrl={analyzedUrl}
          yieldDisplay={yieldNum.toFixed(1)}
          gaugeOffset={gaugeOffset}
          score={score}
          macro={macro}
          onReset={onReset}
          onOpenScenario={() => setScenarioOpen(true)}
        />
      )}

      {scenarioOpen && (
        <ScenarioModal
          listingYieldPct={
            score ? score.scoring.actual_gross_yield * 100 : null
          }
          onClose={() => setScenarioOpen(false)}
        />
      )}
      {macroOpen && (
        <MacroOverlay macro={macro} onClose={() => setMacroOpen(false)} />
      )}
    </div>
  );
}
