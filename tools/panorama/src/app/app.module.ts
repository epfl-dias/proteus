import { BrowserModule } from '@angular/platform-browser';
import { NgModule } from '@angular/core';

import { AppRoutingModule } from './app-routing.module';
import { AppComponent } from './app.component';
import { TimelineComponent } from './timeline/timeline.component';
import { BwProfileComponent } from './bw-profile/bw-profile.component';
import { OverviewComponent } from './overview/overview.component';
import { TimelineRangeComponent } from './timeline-range/timeline-range.component';

@NgModule({
  declarations: [
    AppComponent,
    TimelineComponent,
    BwProfileComponent,
    OverviewComponent,
    TimelineRangeComponent
  ],
  imports: [
    BrowserModule,
    AppRoutingModule
  ],
  providers: [],
  bootstrap: [AppComponent]
})
export class AppModule { }
