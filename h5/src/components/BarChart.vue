<script setup lang="ts">
import { computed, nextTick, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import * as echarts from 'echarts/core'
import { BarChart } from 'echarts/charts'
import { GridComponent, TooltipComponent } from 'echarts/components'
import { CanvasRenderer } from 'echarts/renderers'
import type { ECharts, EChartsCoreOption } from 'echarts/core'

echarts.use([BarChart, GridComponent, TooltipComponent, CanvasRenderer])

const props = defineProps<{
  title: string
  labels: string[]
  values: number[]
  valueSuffix?: string
}>()

const chartEl = ref<HTMLDivElement | null>(null)
let chart: ECharts | null = null

const option = computed<EChartsCoreOption>(() => ({
  tooltip: {
    trigger: 'axis',
    valueFormatter: (value: unknown) => `${value}${props.valueSuffix ?? ''}`,
  },
  grid: { left: 34, right: 12, top: 22, bottom: 28 },
  xAxis: {
    type: 'category',
    data: props.labels,
    axisTick: { show: false },
    axisLine: { lineStyle: { color: '#bfdbfe' } },
    axisLabel: { color: '#64748b', fontSize: 11 },
  },
  yAxis: {
    type: 'value',
    axisLabel: { color: '#64748b', fontSize: 11 },
    splitLine: { lineStyle: { color: '#e0f2fe' } },
  },
  series: [
    {
      name: props.title,
      type: 'bar',
      data: props.values,
      barMaxWidth: 26,
      itemStyle: {
        borderRadius: [8, 8, 0, 0],
        color: '#2563eb',
      },
    },
  ],
}))

function renderChart() {
  if (!chartEl.value) return
  chart ??= echarts.init(chartEl.value)
  chart.setOption(option.value, true)
}

function handleResize() {
  chart?.resize()
}

onMounted(async () => {
  await nextTick()
  renderChart()
  window.addEventListener('resize', handleResize)
})

watch(option, () => renderChart(), { deep: true })

onBeforeUnmount(() => {
  window.removeEventListener('resize', handleResize)
  chart?.dispose()
  chart = null
})
</script>

<template>
  <div class="chart-card">
    <div class="chart-title">{{ title }}</div>
    <div ref="chartEl" class="chart-canvas" />
  </div>
</template>
