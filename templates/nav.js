class NavigationDropdown extends HTMLElement {
    constructor() {
      super();

      // Get the initial expanded state from the attribute, default to false
      const initialExpanded = this.getAttribute('expanded') === 'true';

      // This script is served both from the site root (index.html) and from
      // chapter pages under c/, so resolve links relative to the current page
      const base = /\/c\/[^/]+\.html$/.test(window.location.pathname) ? '../' : './';

      this.innerHTML = `
        <div>
          <button class="dropdown-button" aria-expanded="${initialExpanded}">
            <span><strong>导航</strong></span>
            <svg class="chevron" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <path d="M19 9l-7 7-7-7" stroke-linecap="round" stroke-linejoin="round"/>
            </svg>
          </button>

          <div class="dropdown-content${initialExpanded ? ' open' : ''}">
    <nav class="chapter-nav">
      <div class="section">
        <h3>链接</h3>
        <ul>
          <li><a href="${base}index.html">首页</a></li>
          <li><a href="https://github.com/jweihe/RLHF-book-Chinese">GitHub 仓库</a></li>
          <li><a href="${base}book.pdf">PDF</a> / <a href="https://arxiv.org/abs/2504.12501"> Arxiv </a></li>
          <li class="inactive">订购纸质版（即将上线）</li>
        </ul>
      </div>

      <div class="section">
        <h3>导论</h3>
        <ol start="1">
          <li><a href="${base}c/01-introduction.html">引言</a></li>
          <li><a href="${base}c/02-related-works.html">关键相关工作</a></li>
          <li><a href="${base}c/03-setup.html">定义与背景</a></li>
        </ol>
      </div>

      <div class="section">
        <h3>问题设定与背景</h3>
        <ol start="4">
          <li><a href="${base}c/04-optimization.html">训练概览</a></li>
          <li><a href="${base}c/05-preferences.html">偏好的本质</a></li>
          <li><a href="${base}c/06-preference-data.html">偏好数据</a></li>
        </ol>
      </div>

      <div class="section">
        <h3>优化方法</h3>
        <ol start="7">
          <li><a href="${base}c/07-reward-models.html">奖励建模</a></li>
          <li><a href="${base}c/08-regularization.html">正则化</a></li>
          <li><a href="${base}c/09-instruction-tuning.html">指令微调</a></li>
          <li><a href="${base}c/10-rejection-sampling.html">拒绝采样</a></li>
          <li><a href="${base}c/11-policy-gradients.html">策略梯度算法</a></li>
          <li><a href="${base}c/12-direct-alignment.html">直接对齐算法</a></li>
        </ol>
      </div>

      <div class="section">
        <h3>进阶专题</h3>
        <ol start="13">
          <li><a href="${base}c/13-cai.html">宪法AI与AI反馈</a></li>
          <li><a href="${base}c/14-reasoning.html">推理训练与推理时扩展</a></li>
          <li><a href="${base}c/15-synthetic.html">合成数据与蒸馏</a></li>
          <li><a href="${base}c/16-evaluation.html">评测</a></li>
        </ol>
      </div>

      <div class="section">
        <h3>开放问题</h3>
        <ol start="17">
          <li><a href="${base}c/17-over-optimization.html">过度优化</a></li>
          <li><a href="${base}c/18-style.html">风格与信息</a></li>
          <li><a href="${base}c/19-character.html">产品、用户体验与模型个性</a></li>
        </ol>
      </div>
    </nav>
  </div>
</div>
      `;

      // Set up click handler
      const button = this.querySelector('.dropdown-button');
      const content = this.querySelector('.dropdown-content');

      button.addEventListener('click', () => {
        const isExpanded = button.getAttribute('aria-expanded') === 'true';
        button.setAttribute('aria-expanded', !isExpanded);
        content.classList.toggle('open');
      });
    }

    // Add attribute change observer
    static get observedAttributes() {
      return ['expanded'];
    }

    attributeChangedCallback(name, oldValue, newValue) {
      if (name === 'expanded') {
        const button = this.querySelector('.dropdown-button');
        const content = this.querySelector('.dropdown-content');
        const isExpanded = newValue === 'true';

        if (button && content) {
          button.setAttribute('aria-expanded', isExpanded);
          content.classList.toggle('open', isExpanded);
        }
      }
    }
}

// Only define the component once
if (!customElements.get('navigation-dropdown')) {
  customElements.define('navigation-dropdown', NavigationDropdown);
}
