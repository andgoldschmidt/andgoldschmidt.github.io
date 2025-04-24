<template>
    <div class="project-card">
      <img v-if="img" :src="img" class="project-img" />
      <div class="project-info">
        <h3 class="project-title">
          <a v-if="link" :href="link" target="_blank" rel="noopener">{{ title }}</a>
          <span v-else>{{ title }}</span>
        </h3>
        <div class="project-desc">
            <slot />
        </div>
      </div>
    </div>
  </template>
  
  <script setup>
  defineProps({
    title: String,
    img: String,
    link: String
  })
  </script>
  