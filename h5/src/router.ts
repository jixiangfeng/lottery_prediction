import { createRouter, createWebHistory, type RouteRecordRaw } from 'vue-router'

export const navItems = [
  { label: '首页', to: '/', icon: '⌂' },
  { label: '推荐', to: '/candidates', icon: '★' },
  { label: '回测', to: '/backtest', icon: '↺' },
  { label: '策略', to: '/strategy', icon: '◇' },
] as const

export const routes: RouteRecordRaw[] = [
  { path: '/', name: 'home', component: () => import('./views/HomeView.vue') },
  { path: '/report/:issue', name: 'report-detail', component: () => import('./views/ReportDetailView.vue') },
  { path: '/candidates', name: 'candidates', component: () => import('./views/CandidatesView.vue') },
  { path: '/backtest', name: 'backtest', component: () => import('./views/BacktestView.vue') },
  { path: '/strategy', name: 'strategy', component: () => import('./views/StrategyView.vue') },
]

export const router = createRouter({
  history: createWebHistory(),
  routes,
})
