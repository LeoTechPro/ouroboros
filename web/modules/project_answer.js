/**
 * Project lifecycle rows in Main (docs/DESIGN.md "Project completion mirror").
 *
 * A Project root that ended with Ouroboros's own final answer reaches Main as an
 * ordinary Ouroboros message: folded when it is long, with the Project chip under
 * it — the chip names the Project and opens it. Every other lifecycle row (a
 * start, an ending with no answer, a row written before the answer rode the row)
 * keeps its System text and the shared `Open Project` action.
 */
import { createSystemMessageAction, createSystemMessageActions, renderProjectChip } from './ui_helpers.js';

function openProject(projectId, projectName) {
    window.dispatchEvent(new CustomEvent('ouro:open-project', {
        detail: { project: { id: projectId, name: projectName || 'Project' } },
    }));
}

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
    const open = () => openProject(projectId, projectName);
    let control;
    if (role === 'assistant') {
        const name = projectName || 'Project';
        control = renderProjectChip({ name, status: '↗', className: 'chat-quiz-project', onClick: open });
        control.title = `Open ${name}`;
        control.querySelector('.chat-live-project-status')?.setAttribute('aria-hidden', 'true');
        bubble.classList.add('project-answer');
        watchFolds();
        afterLayout(() => markFold(bubble));
    } else {
        control = createSystemMessageAction({ label: 'Open Project ↗', onClick: open });
    }
    const actions = createSystemMessageActions(control);
    bubble.querySelector('.message')?.after(actions);
    return actions;
}
