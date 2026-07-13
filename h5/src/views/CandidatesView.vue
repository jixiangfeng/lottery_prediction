<script setup lang="ts">
import NumberBall from '../components/NumberBall.vue'
import ReportShell from '../components/ReportShell.vue'
import type { ReportData } from '../domain/report'

function sortedCandidates(report: ReportData) {
  return [...report.candidateGroups].sort((left, right) => right.score - left.score)
}

function formatNumbers(numbers?: number[] | null) {
  return numbers?.map((number) => String(number).padStart(2, '0')).join(' ') || '--'
}

function formatPositionPools(pools?: Record<string, number[]> | null) {
  if (!pools) return ''
  return Object.entries(pools)
    .map(([position, digits]) => `${position}: ${digits.join(',')}`)
    .join('；')
}
</script>

<template>
  <ReportShell v-slot="{ report }">
    <section class="panel">
      <header>
        <div>
          <p class="eyebrow">Candidates</p>
          <h2>选十候选组</h2>
        </div>
        <span class="pill">按评分排序</span>
      </header>
      <div class="candidate-list">
        <article v-for="group in sortedCandidates(report)" :key="group.rank" class="candidate-card">
          <div class="candidate-head">
            <strong>#{{ group.rank }}</strong>
            <span>评分 {{ group.score.toFixed(2) }}</span>
          </div>
          <div class="balls compact">
            <NumberBall v-for="num in group.numbers" :key="num" :value="num" />
          </div>
          <div class="tags">
            <span>奇偶 {{ group.oddCount }}:{{ group.evenCount }}</span>
            <span>大小 {{ group.bigCount }}:{{ group.smallCount }}</span>
            <span>上期重号 {{ group.repeatLastCount }}</span>
          </div>
        </article>
      </div>
      <div v-if="report.candidateCoverage" class="summary-grid coverage-grid">
        <article class="metric-card">
          <span>覆盖号码</span>
          <strong>{{ report.candidateCoverage.totalUniqueNumbers }}</strong>
        </article>
        <article class="metric-card">
          <span>平均重合</span>
          <strong>{{ report.candidateCoverage.averageOverlap }}</strong>
        </article>
        <article class="metric-card">
          <span>最大重合</span>
          <strong>{{ report.candidateCoverage.maxOverlap }}</strong>
        </article>
      </div>
      <div v-if="report.candidatePortfolioScore" class="summary-grid coverage-grid">
        <article class="metric-card">
          <span>组合总评分</span>
          <strong>{{ report.candidatePortfolioScore.finalScore }}</strong>
        </article>
        <article class="metric-card">
          <span>评级</span>
          <strong>{{ report.candidatePortfolioScore.grade }}</strong>
        </article>
        <article class="metric-card">
          <span>低重合分</span>
          <strong>{{ report.candidatePortfolioScore.overlapScore }}</strong>
        </article>
        <article v-if="report.candidateBatchOptimization" class="metric-card">
          <span>优化批次</span>
          <strong>{{ report.candidateBatchOptimization.trialCount }}</strong>
        </article>
      </div>

      <section v-if="report.bettingPlan" class="betting-panel">
        <header>
          <div>
            <p class="eyebrow">Betting Plan</p>
            <h2>复式/胆拖建议</h2>
          </div>
          <span class="pill">{{ report.bettingPlan.displayName }}</span>
        </header>
        <div class="tags">
          <span>核心 {{ formatNumbers(report.bettingPlan.coreNumbers) }}</span>
          <span>辅助 {{ formatNumbers(report.bettingPlan.assistNumbers) }}</span>
        </div>
        <div class="strategy-list betting-list">
          <article v-for="item in report.bettingPlan.plans" :key="`${item.kind}-${item.title}`" class="strategy-row strategy-row-dense">
            <span>{{ item.kind }} · {{ item.title }}</span>
            <strong :class="{ loss: item.riskLevel === '高' }">{{ item.cost ? `${item.cost}元` : '需复核' }}</strong>
            <em>{{ item.positionPools ? formatPositionPools(item.positionPools) : item.bankerNumbers ? `胆 ${formatNumbers(item.bankerNumbers)}；拖 ${formatNumbers(item.dragNumbers)}` : formatNumbers(item.numbers) }}</em>
            <em>注数 {{ item.betCount || '--' }} · 风险 {{ item.riskLevel }} · {{ item.note }}</em>
          </article>
        </div>
        <p class="hint">{{ report.bettingPlan.disclaimer }}</p>
      </section>
    </section>
  </ReportShell>
</template>
