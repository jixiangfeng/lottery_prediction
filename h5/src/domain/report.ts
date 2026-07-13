export interface CandidateGroup {
  rank: number
  numbers: number[]
  score: number
  oddCount: number
  evenCount: number
  bigCount: number
  smallCount: number
  repeatLastCount: number
  zoneDistribution: number[]
}

export interface BacktestSummary {
  totalCost: number
  totalPrize: number
  roi: number
  roiPercent?: string
  averageHit: number
  hitDistribution?: Record<string, number>
  maxMissStreak?: number
}

export interface StrategyResult {
  strategy: string
  summary?: BacktestSummary | null
}

export interface WalkForwardWeightMeta {
  enabled: boolean
  alpha: number
  matchedCount?: number
  bestStrategy?: string
  strategyScores?: Record<string, number>
  normalizedScores?: Record<string, number>
}

export interface WalkForwardStrategySummary {
  strategy: string
  periodCount: number
  groupCount: number
  totalCost: number
  totalPrize: number
  cappedTotalPrize?: number
  roi: number
  cappedRoi?: number
  averageHit: number
  hit5PlusCount: number
  hit6PlusCount: number
  issueHit5PlusCount: number
  maxLosingStreak: number
  bestIssue: string
  bestIssueMaxHit: number
  recentRoi: number
  score: number
}

export interface WalkForwardStrategyReport {
  schemaVersion: number
  lottery: string
  play: string
  periodCount: number
  minTrainSize: number
  groupCount: number
  ticketPrice: number
  bestStrategy: string
  summaries: WalkForwardStrategySummary[]
  disclaimer?: string
}

export interface BettingPlanItem {
  kind: string
  title: string
  numbers?: number[] | null
  numberCount?: number | null
  bankerNumbers?: number[] | null
  dragNumbers?: number[] | null
  positionPools?: Record<string, number[]> | null
  betCount: number
  cost: number
  riskLevel: string
  note: string
}

export interface BettingPlan {
  play: string
  displayName: string
  coreNumbers: number[]
  assistNumbers: number[]
  defensiveNumbers: number[]
  budgets: number[]
  plans: BettingPlanItem[]
  disclaimer?: string
}

export interface ReportData {
  schemaVersion: number
  lottery: string
  play: string
  issue: string
  parameterName: string
  strategyMode?: string
  latestNumbers: number[]
  hotNumbers?: number[]
  coldNumbers?: number[]
  candidateGroups: CandidateGroup[]
  backtest?: BacktestSummary | null
  strategyComparison?: Record<string, StrategyResult>
  slidingWindow?: Record<string, unknown>
  parameterSearch?: Array<Record<string, unknown>>
  candidateCoverage?: {
    groupCount: number
    totalUniqueNumbers: number
    averageOverlap: number
    maxOverlap: number
    zoneCoverageCount: number
    tailCoverageCount: number
  } | null
  candidatePortfolioScore?: {
    finalScore: number
    grade: string
    coverageScore: number
    overlapScore: number
    maxOverlapScore: number
  } | null
  candidateBatchOptimization?: {
    enabled: boolean
    trialCount: number
    bestIndex: number
  } | null
  walkForwardValidation?: {
    enabled: boolean
    bestParameter?: string | null
    trainWindowCount: number
    testWindowCount: number
    rows: Array<{
      parameter: string
      trainMeanRoi: number
      testMeanRoi: number
      generalizationGap: number
      testWinRate: number
      riskLevel: string
      score: number
    }>
  } | null
  walkForwardParameterWeights?: WalkForwardWeightMeta | null
  bettingPlan?: BettingPlan | null
  artifacts?: {
    html?: string | null
    pickSnapshot?: string | null
  }
  dataQuality?: {
    ok: boolean
    totalIssues: number
    latestIssue?: string | null
    errorCount: number
    warningCount: number
  } | null
  dataSource?: {
    status: string
    mode: string
    latestIssue: string
    totalIssues: number
    usedCache: boolean
    updated: boolean
    source: string
  } | null
  liveParameterWeights?: {
    enabled: boolean
    alpha: number
    matchedCount?: number
  } | null
  disclaimer?: string
}

function percent(value: number | undefined): string {
  if (typeof value !== 'number' || Number.isNaN(value)) {
    return '--'
  }
  return `${(value * 100).toFixed(2)}%`
}

function normalizeBacktest(summary?: BacktestSummary | null): BacktestSummary | null {
  if (!summary) {
    return null
  }
  return {
    ...summary,
    roiPercent: percent(summary.roi),
  }
}

export function normalizeReportData(raw: unknown): ReportData {
  const data = raw as ReportData
  return {
    ...data,
    latestNumbers: data.latestNumbers ?? [],
    candidateGroups: data.candidateGroups ?? [],
    strategyComparison: data.strategyComparison ?? {},
    slidingWindow: data.slidingWindow ?? {},
    parameterSearch: data.parameterSearch ?? [],
    walkForwardParameterWeights: data.walkForwardParameterWeights ?? null,
    bettingPlan: data.bettingPlan ?? null,
    backtest: normalizeBacktest(data.backtest),
  }
}

export function normalizeWalkForwardStrategyReport(raw: unknown): WalkForwardStrategyReport {
  const data = raw as WalkForwardStrategyReport
  return {
    ...data,
    summaries: data.summaries ?? [],
  }
}

export function topStrategies(report: ReportData): StrategyResult[] {
  return Object.values(report.strategyComparison ?? {})
    .map((item) => ({ ...item, summary: normalizeBacktest(item.summary) }))
    .sort((left, right) => (right.summary?.roi ?? -Infinity) - (left.summary?.roi ?? -Infinity))
}

export function topWalkForwardSummaries(report?: WalkForwardStrategyReport | null): WalkForwardStrategySummary[] {
  return [...(report?.summaries ?? [])].sort((left, right) => right.score - left.score)
}

export function numberZone(number: number): string {
  if (number <= 20) return 'zone-a'
  if (number <= 40) return 'zone-b'
  if (number <= 60) return 'zone-c'
  return 'zone-d'
}
