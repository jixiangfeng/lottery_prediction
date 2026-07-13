<script setup lang="ts">
import { onMounted, ref } from 'vue'
import BarChart from '../components/BarChart.vue'
import ReportShell from '../components/ReportShell.vue'
import { fetchWalkForwardStrategyReport } from '../api/report'
import { buildStrategyRoiSeries } from '../domain/charts'
import { topStrategies, topWalkForwardSummaries, type WalkForwardStrategyReport } from '../domain/report'

const asNumber = (value: unknown) => (typeof value === 'number' ? value : Number(value))
const percent = (value: number | undefined) => (typeof value === 'number' ? `${(value * 100).toFixed(2)}%` : '--')
const walkForward = ref<WalkForwardStrategyReport | null>(null)
const walkForwardError = ref('')

onMounted(async () => {
  try {
    walkForward.value = await fetchWalkForwardStrategyReport()
  } catch (error) {
    walkForwardError.value = error instanceof Error ? error.message : '读取前推回测失败'
  }
})
</script>

<template>
  <ReportShell v-slot="{ report }">
    <section class="panel">
      <header>
        <div>
          <p class="eyebrow">Strategies</p>
          <h2>策略对比</h2>
        </div>
      </header>
      <BarChart
        title="策略收益率"
        :labels="buildStrategyRoiSeries(report).labels"
        :values="buildStrategyRoiSeries(report).values"
        value-suffix="%"
      />
      <div class="strategy-list">
        <article v-for="item in topStrategies(report)" :key="item.strategy" class="strategy-row">
          <span>{{ item.strategy }}</span>
          <strong :class="{ loss: (item.summary?.roi ?? 0) < 0 }">{{ item.summary?.roiPercent }}</strong>
          <em>返奖 {{ item.summary?.totalPrize ?? '--' }}</em>
        </article>
      </div>
    </section>

    <section class="panel">
      <header>
        <div>
          <p class="eyebrow">Parameter Search</p>
          <h2>参数搜索排名</h2>
        </div>
      </header>
      <div class="param-list">
        <article v-for="item in report.parameterSearch" :key="String(item.name)" class="strategy-row">
          <span>#{{ item.rank }} {{ item.name }}</span>
          <strong>{{ asNumber(item.score).toFixed(4) }}</strong>
        </article>
      </div>
    </section>

    <section v-if="report.walkForwardParameterWeights" class="panel">
      <header>
        <div>
          <p class="eyebrow">Daily Weighting</p>
          <h2>前推加权状态</h2>
        </div>
        <span class="pill">{{ report.walkForwardParameterWeights.enabled ? '已启用' : '未启用' }}</span>
      </header>
      <div class="summary-grid compact">
        <article class="metric-card">
          <span>前推最稳</span>
          <strong>{{ report.walkForwardParameterWeights.bestStrategy ?? '--' }}</strong>
        </article>
        <article class="metric-card">
          <span>加权强度</span>
          <strong>{{ report.walkForwardParameterWeights.alpha }}</strong>
        </article>
        <article class="metric-card">
          <span>匹配策略</span>
          <strong>{{ report.walkForwardParameterWeights.matchedCount ?? 0 }}</strong>
        </article>
      </div>
    </section>

    <section v-if="walkForward" class="panel">
      <header>
        <div>
          <p class="eyebrow">Walk-forward Strategy Backtest</p>
          <h2>逐期前推策略表现</h2>
        </div>
        <span class="pill">{{ walkForward.periodCount }} 期 · 最稳 {{ walkForward.bestStrategy }}</span>
      </header>
      <div class="strategy-list">
        <article v-for="item in topWalkForwardSummaries(walkForward)" :key="item.strategy" class="strategy-row strategy-row-dense">
          <span>{{ item.strategy }}</span>
          <strong :class="{ loss: (item.cappedRoi ?? item.roi) < 0 }">{{ percent(item.cappedRoi ?? item.roi) }}</strong>
          <em>均命中 {{ item.averageHit.toFixed(3) }} · 中5+ {{ item.hit5PlusCount }} · 连亏 {{ item.maxLosingStreak }}</em>
        </article>
      </div>
      <p class="hint">封顶ROI按单注最高 800 元统计，降低偶发大奖对策略排名的干扰；仅供历史模拟参考。</p>
    </section>
    <section v-else-if="walkForwardError" class="panel state-card error">{{ walkForwardError }}</section>

    <section v-if="report.walkForwardValidation" class="panel">
      <header>
        <div>
          <p class="eyebrow">Walk Forward</p>
          <h2>反过拟合前推验证</h2>
        </div>
        <span class="pill">最稳 {{ report.walkForwardValidation.bestParameter }}</span>
      </header>
      <div class="strategy-list">
        <article v-for="item in report.walkForwardValidation.rows" :key="item.parameter" class="strategy-row">
          <span>{{ item.parameter }}</span>
          <strong :class="{ loss: item.riskLevel === '高' }">{{ item.riskLevel }}</strong>
          <em>测试 {{ (item.testMeanRoi * 100).toFixed(2) }}%</em>
        </article>
      </div>
    </section>
  </ReportShell>
</template>
