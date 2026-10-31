// Configurar antes de carregar MathJax e renderizar novamente ao navegar no site.
window.MathJax = {
  tex: {
    inlineMath: [["\\(", "\\)"]],
    displayMath: [["\\[", "\\]"]],
    processEscapes: true,
    processEnvironments: true
  },
  options: {
    ignoreHtmlClass: ".*|",
    processHtmlClass: "arithmatex"
  },
  startup: {
    typeset: false,
    ready: () => {
      MathJax.startup.defaultReady();
      MathJax.startup.promise.then(() => {
        let pending = Promise.resolve();
        const render = () => {
          pending = pending.then(() => {
            MathJax.typesetClear();
            return MathJax.typesetPromise();
          }).catch(error => console.error("Erro ao renderizar fórmulas:", error));
        };
        render();
        if (typeof document$ !== "undefined") document$.subscribe(render);
      });
    }
  }
};
