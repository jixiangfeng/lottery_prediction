<script setup lang="ts">
import { computed, onMounted, watch } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import NumberBall from '../components/NumberBall.vue'
import { fetchReportCatalog } from '../api/catalog'
import { useReportStore } from '../stores/reportStore'
import { selectReportIssue } from '../domain/catalog'

const route = useRoute()
const router = useRouter()
const { state, loadReport } = useReportStore()

async function syncIssue(issue?: string) {
  await loadReport(issue || undefined, true)
}

const currentIssue = computed(() => route.params.issue as string | undefined)

async function copyShareLink() {
  const issue = state.report?.issue ?? currentIssue.value
  if (!issue) return
  const link = `${window.location.origin}/report/${issue}`
  await navigator.clipboard.writeText(link)
}

function goHome() {
  void router.push({ name: 'home' })
}

onMounted(async () => {
  const catalog = await fetchReportCatalog()
  const issue = selectReportIssue(catalog, currentIssue.value)
  await syncIssue(issue)
  if (!currentIssue.value || currentIssue.value !== issue) {
    await router.replace({ name: 'report-detail', params: { issue } })
  }
})

watch(
  () => currentIssue.value,
  (issue) => {
    if (issue) void syncIssue(issue)
  },
)
</script>

<template>
  <section v-if="state.loading" class="state-card">正在加载日报详情...</section>
  <section v-else-if="state.error" class="state-card error">{{ state.error }}</section>
  <template v-else-if="state.report">
    <section class="hero-card blue-hero">
      <p class="eyebrow">快乐8 H5 用户端</p>
      <h1>日报详情</h1>
      <p class="subline">期号 {{ state.report.issue }} · 支持独立分享链接</p>
    </section>

    <section class="panel">
      <header>
        <div>
          <p class="eyebrow">Latest Draw</p>
          <h2>最新开奖号码</h2>
        </div>
        <div class="header-actions">
          <button class="pill" @click="copyShareLink">复制链接</button>
          <button class="pill" @click="goHome">返回首页</button>
        </div>
      </header>
      <div class="balls">
        <NumberBall v-for="num in state.report.latestNumbers" :key="num" :value="num" />
      </div>
    </section>

    <section class="summary-grid">
      <article class="metric-card">
        <span>主推荐参数</span>
        <strong>{{ state.report.parameterName }}</strong>
      </article>
      <article class="metric-card">
        <span>策略模式</span>
        <strong>{{ state.report.strategyMode ?? 'auto' }}</strong>
      </article>
      <article class="metric-card">
        <span>最近回测 ROI</span>
        <strong :class="{ loss: (state.report.backtest?.roi ?? 0) < 0 }">{{ state.report.backtest?.roiPercent }}</strong>
      </article>
      <article class="metric-card">
        <span>候选组数量</span>
        <strong>{{ state.report.candidateGroups.length }}</strong>
      </article>
      <article class="metric-card">
        <span>数据质量</span>
        <strong :class="{ loss: state.report.dataQuality?.ok === false }">
          {{ state.report.dataQuality?.ok === false ? '异常' : '正常' }}
        </strong>
      </article>
    </section>
  </template>
</template>
