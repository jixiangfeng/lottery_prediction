<script setup lang="ts">
import BarChart from '../components/BarChart.vue'
import ReportShell from '../components/ReportShell.vue'
import { buildHitDistributionSeries } from '../domain/charts'
</script>

<template>
  <ReportShell v-slot="{ report }">
    <section class="panel">
      <header>
        <div>
          <p class="eyebrow">Backtest</p>
          <h2>最近100期回测</h2>
        </div>
      </header>
      <div class="summary-grid">
        <article class="metric-card">
          <span>投入</span>
          <strong>{{ report.backtest?.totalCost ?? '--' }}</strong>
        </article>
        <article class="metric-card">
          <span>返奖</span>
          <strong>{{ report.backtest?.totalPrize ?? '--' }}</strong>
        </article>
        <article class="metric-card">
          <span>收益率</span>
          <strong :class="{ loss: (report.backtest?.roi ?? 0) < 0 }">{{ report.backtest?.roiPercent }}</strong>
        </article>
      </div>
      <BarChart
        title="命中分布"
        :labels="buildHitDistributionSeries(report).labels"
        :values="buildHitDistributionSeries(report).values"
      />
      <div class="hit-grid">
        <article v-for="(count, hit) in report.backtest?.hitDistribution" :key="hit" class="hit-card">
          <span>中{{ hit }}</span>
          <strong>{{ count }}</strong>
        </article>
      </div>
    </section>
  </ReportShell>
</template>
