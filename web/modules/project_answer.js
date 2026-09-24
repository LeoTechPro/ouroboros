/**
 * Project lifecycle rows in Main (docs/DESIGN.md "Project completion mirror").
 *
 * A Project root that ended with Ouroboros's own final answer reaches Main as an
 * ordinary Ouroboros message, folded when it is long. Every other lifecycle row (a
 * start, an ending with no answer, a row written before the answer rode the row)
 * keeps its System text. Either way the row ends with the same Project reference:
 * the voice of a row never chooses how the UI points at its Project.
 */
import { createSystemMessageActions } from './ui_helpers.js';
import { projectReference } from './project_reference.js';

// The fade says "there is more": it appears only when the fold really hides text. The clamp
// itself is unconditional CSS, so toggling the class never moves layout. A row mounted while
// Chat is not the visible page has no layout box (clientHeight 0): it is left alone and
// measured when Chat is shown; a resize re-measures, because the width decides the height.
function markFold(bubble) {
    const message = bubble.querySelector('.message');
    if (!message || !message.clientHeight) return;
    bubble.classList.toggle('is-folded', message.scrollHeight > message.clientHeight + 1);
}

// One pair of window listeners for the app's lifetime, however many rows exist: nothing is
// held per row, so a released row needs no disposer here.
let watching = false;
let queued = false;
function afterLayout(work) {
    if (typeof requestAnimationFrame === 'function') requestAnimationFrame(work);
}
function refreshFoldsSoon() {
    if (queued) return;
    queued = true;
    afterLayout(() => {
        queued = false;
        document.querySelectorAll('.chat-bubble.project-answer').forEach(markFold);
    });
}
function watchFolds() {
    if (watching || typeof window === 'undefined') return;
    watching = true;
    window.addEventListener('ouro:page-shown', refreshFoldsSoon);
    window.addEventListener('resize', refreshFoldsSoon);
}

export function decorateProjectRow(bubble, { role = 'system', projectId = '', projectName = '' } = {}) {
    if (!bubble || !projectId) return null;
    if (role === 'assistant') {
        bubble.classList.add('project-answer');
        watchFolds();
        afterLayout(() => markFold(bubble));
    }
    const actions = createSystemMessageActions(projectReference({ id: projectId, name: projectName }));
    bubble.querySelector('.message')?.after(actions);
    return actions;
}
